from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from f1_strategy.evaluation import EvaluationReport, render_markdown, report_to_dict
from f1_strategy.metadata import APP_VERSION
from f1_strategy.models import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    feature_schema_hash,
    model_manifest,
    model_manifest_path,
)
from f1_strategy.deployment import load_registry, rollback_candidate
from f1_strategy.replay import (
    ReplayEvaluationReport,
    ReplaySuiteReport,
    run_benchmark_replay_suite,
    replay_report_to_dict,
    replay_suite_to_dict,
)


DEFAULT_ARTIFACT_ROOT = "artifacts/models"
LOCAL_MODEL_PATHS = {
    "xgboost": "models/xgboost_lap_delta.json",
    "lightgbm": "models/lightgbm_lap_delta.txt",
    "catboost": "models/catboost_lap_delta.cbm",
    "sequence": "models/sequence_lap_delta.pt",
}


@dataclass(frozen=True)
class ArtifactBundle:
    artifact_id: str
    bundle_dir: Path
    model_path: Path
    manifest_path: Path
    registry_path: Path


@dataclass(frozen=True)
class PromotionGateConfig:
    max_mean_mae_lap_delta_s: float = 1.25
    min_mean_coverage_pct: float = 50.0
    max_calibration_error_pct: float = 20.0
    max_mean_interval_width_s: float = 5.5
    max_latency_p95_ms: float = 25.0
    max_monotonic_wear_violations: int = 0
    # Benchmark-suite splits are controlled scenarios; strict 0.35s threshold.
    max_replay_mae_lap_delta_s: float = 0.35
    # Individual dataset replay uses real-world data (SC/VSC laps inflate MAE and
    # stint-boundary resets can introduce transient wear drops); looser bounds.
    max_real_replay_mae_lap_delta_s: float = 2.0
    max_real_replay_monotonic_wear_violations: int = 2
    min_replay_coverage_pct: float = 80.0
    max_replay_missing_target_pct: float = 0.0
    min_replay_event_count: int = 12
    max_pit_target_error_laps: float = 7.0
    max_strategy_regret_s: float = 2.5
    require_replay_evaluation: bool = True
    require_replay_suite: bool = True
    required_replay_suite_name: str = "benchmark"
    min_replay_suite_split_count: int = 5
    require_production_replay_validation: bool = False


@dataclass(frozen=True)
class PromotionResult:
    artifact_id: str
    promoted: bool
    failures: list[str]


@dataclass(frozen=True)
class RegisteredArtifact:
    backend: str
    model_path: Path
    bundle: ArtifactBundle | None
    skipped_reason: str | None = None


@dataclass(frozen=True)
class PruneResult:
    archived: list[str]
    kept: list[str]


def artifact_release_detail(
    artifact_id: str,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    gates: PromotionGateConfig | None = None,
) -> dict[str, Any]:
    if ".." in Path(artifact_id).parts:
        raise RuntimeError(f"Invalid artifact id: {artifact_id}")
    root = Path(artifact_root)
    bundle_dir = root / artifact_id
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Model artifact manifest does not exist: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = _read_optional_json(bundle_dir / str(manifest.get("evaluation_report", "")))
    replay = _read_optional_json(bundle_dir / str(manifest.get("replay_evaluation_report", "")))
    replay_suite = _read_optional_json(bundle_dir / str(manifest.get("replay_suite_report", "")))
    failures = promotion_gate_failures(
        manifest=manifest,
        bundle_dir=bundle_dir,
        gates=gates or PromotionGateConfig(),
    )
    registry = load_registry(root)
    return {
        "artifact_id": artifact_id,
        "bundle_dir": str(bundle_dir),
        "manifest": manifest,
        "evaluation": evaluation,
        "replay_evaluation": replay,
        "replay_suite": replay_suite,
        "promotion_failures": failures,
        "promotion_ready": not failures,
        "promoted": is_registry_promoted(registry, artifact_id),
    }


def register_existing_model_artifact(
    model_path: str | Path,
    backend: str,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    replay_dataset: str | Path = "examples/replay_telemetry.csv",
    promote: bool = False,
) -> ArtifactBundle:
    from f1_strategy.engine import InferenceEngine
    from f1_strategy.evaluation import run_evaluation
    from f1_strategy.models import ModelConfig, create_serving_model
    from f1_strategy.replay import run_replay_evaluation

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact does not exist: {path}")

    serving_model = create_serving_model(
        config=ModelConfig(),
        backend=backend,
        xgboost_model_path=str(path) if backend == "xgboost" else LOCAL_MODEL_PATHS["xgboost"],
        lightgbm_model_path=str(path) if backend == "lightgbm" else LOCAL_MODEL_PATHS["lightgbm"],
        catboost_model_path=str(path) if backend == "catboost" else LOCAL_MODEL_PATHS["catboost"],
        sequence_model_path=str(path) if backend == "sequence" else LOCAL_MODEL_PATHS["sequence"],
    )
    model_paths = {backend: str(path)}
    if backend == "sequence":
        model_paths["sequence"] = str(path)
    evaluation_report = run_evaluation(model_backend=backend, model_paths=model_paths)
    replay_report = run_replay_evaluation(
        replay_dataset,
        engine=InferenceEngine(model=serving_model),
    )
    from dataclasses import replace as _replace
    from f1_strategy.config import load_settings

    _backend = backend
    _path = str(path)
    replay_suite = run_benchmark_replay_suite(
        engine_factory=lambda base: InferenceEngine(
            model=create_serving_model(
                config=ModelConfig(base_lap_time_s=base),
                backend=_backend,
                xgboost_model_path=_path if _backend == "xgboost" else LOCAL_MODEL_PATHS["xgboost"],
                lightgbm_model_path=_path if _backend == "lightgbm" else LOCAL_MODEL_PATHS["lightgbm"],
                catboost_model_path=_path if _backend == "catboost" else LOCAL_MODEL_PATHS["catboost"],
                sequence_model_path=_path if _backend == "sequence" else LOCAL_MODEL_PATHS["sequence"],
            ),
            settings=_replace(load_settings(), base_lap_time_s=base),
        )
    )
    return create_model_artifact_bundle(
        model_path=path,
        backend=backend,
        training_config={
            "backend": backend,
            "output": str(path),
            "source": "registered-existing-local-model",
            "training_rows": _training_rows_from_manifest(path),
            "feature_schema_hash": feature_schema_hash(),
        },
        evaluation_report=evaluation_report,
        replay_evaluation_report=replay_report,
        replay_suite_report=replay_suite,
        artifact_root=artifact_root,
        promote=promote,
    )


def register_local_model_artifacts(
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    replay_dataset: str | Path = "examples/replay_telemetry.csv",
    promote: bool = False,
) -> list[RegisteredArtifact]:
    results: list[RegisteredArtifact] = []
    for backend, model_path in LOCAL_MODEL_PATHS.items():
        path = Path(model_path)
        if not path.exists():
            results.append(
                RegisteredArtifact(
                    backend=backend,
                    model_path=path,
                    bundle=None,
                    skipped_reason="model file missing",
                )
            )
            continue
        try:
            bundle = register_existing_model_artifact(
                model_path=path,
                backend=backend,
                artifact_root=artifact_root,
                replay_dataset=replay_dataset,
                promote=promote,
            )
        except Exception as exc:
            results.append(
                RegisteredArtifact(
                    backend=backend,
                    model_path=path,
                    bundle=None,
                    skipped_reason=str(exc),
                )
            )
        else:
            results.append(
                RegisteredArtifact(
                    backend=backend,
                    model_path=path,
                    bundle=bundle,
                )
            )
    return results


def prune_artifact_registry(
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    keep_candidates_per_backend: int = 2,
) -> PruneResult:
    root = Path(artifact_root)
    registry_path = root / "registry.json"
    registry = load_registry(root)
    promoted_ids = set(registry.get("promoted", {}).values())
    candidates_by_backend: dict[str, list[dict[str, Any]]] = {}
    for artifact in registry.get("artifacts", []):
        if artifact.get("status") == "candidate":
            candidates_by_backend.setdefault(str(artifact.get("backend")), []).append(artifact)
    keep_ids = set(promoted_ids)
    for candidates in candidates_by_backend.values():
        candidates.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        keep_ids.update(
            item["artifact_id"] for item in candidates[: max(0, keep_candidates_per_backend)]
        )

    archived: list[str] = []
    kept: list[str] = []
    updated = []
    for artifact in registry.get("artifacts", []):
        item = dict(artifact)
        if item["artifact_id"] in keep_ids:
            kept.append(item["artifact_id"])
        elif item.get("status") == "candidate":
            item["status"] = "archived"
            archived.append(item["artifact_id"])
        updated.append(item)

        manifest_path = root / item["artifact_id"] / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != item.get("status"):
                manifest["status"] = item["status"]
                _write_json(manifest_path, manifest)
                model_filename = manifest.get("model_filename")
                if model_filename:
                    _write_json(model_manifest_path(manifest_path.parent / str(model_filename)), manifest)

    registry["artifacts"] = updated
    _write_json(registry_path, registry)
    return PruneResult(archived=archived, kept=kept)


def create_model_artifact_bundle(
    model_path: str | Path,
    backend: str,
    training_config: dict[str, Any],
    evaluation_report: EvaluationReport,
    replay_evaluation_report: ReplayEvaluationReport | None = None,
    replay_suite_report: ReplaySuiteReport | None = None,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    replay_dataset: str | Path | None = None,
    promote: bool = False,
    created_at: str | None = None,
    git_sha: str | None = None,
) -> ArtifactBundle:
    source_model_path = Path(model_path)
    if not source_model_path.exists():
        raise FileNotFoundError(f"Model artifact does not exist: {source_model_path}")

    timestamp = created_at or _utc_timestamp()
    resolved_git_sha = git_sha or current_git_sha()
    artifact_id = f"{backend}/{timestamp}-{resolved_git_sha}"
    bundle_dir = Path(artifact_root) / backend / f"{timestamp}-{resolved_git_sha}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    bundled_model_path = bundle_dir / source_model_path.name
    shutil.copy2(source_model_path, bundled_model_path)

    evaluation_payload = report_to_dict(evaluation_report)
    replay_payload = (
        replay_report_to_dict(replay_evaluation_report)
        if replay_evaluation_report is not None
        else None
    )
    replay_suite_payload = (
        replay_suite_to_dict(replay_suite_report) if replay_suite_report is not None else None
    )
    manifest = _artifact_manifest(
        artifact_id=artifact_id,
        backend=backend,
        created_at=timestamp,
        git_sha=resolved_git_sha,
        model_filename=bundled_model_path.name,
        training_config=training_config,
        evaluation_payload=evaluation_payload,
        replay_payload=replay_payload,
        replay_suite_payload=replay_suite_payload,
        promoted=promote,
    )
    if replay_dataset is not None:
        manifest.setdefault("training_config", {})
        manifest["training_config"]["replay_dataset_path"] = str(replay_dataset)
        manifest["training_parameters"] = manifest["training_config"]
    model_card_payload = _model_card_payload(
        manifest=manifest,
        evaluation_payload=evaluation_payload,
        replay_payload=replay_payload,
        replay_suite_payload=replay_suite_payload,
    )
    manifest["model_card"] = "model_card.json"

    manifest_path = bundle_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(model_manifest_path(bundled_model_path), manifest)
    _write_json(bundle_dir / "training_config.json", training_config)
    _write_json(bundle_dir / "model_card.json", model_card_payload)
    _write_json(bundle_dir / "evaluation.json", evaluation_payload)
    (bundle_dir / "evaluation.md").write_text(render_markdown(evaluation_report), encoding="utf-8")
    if replay_payload is not None:
        _write_json(bundle_dir / "replay_evaluation.json", replay_payload)
    if replay_suite_payload is not None:
        _write_json(bundle_dir / "replay_suite.json", replay_suite_payload)

    registry_path = Path(artifact_root) / "registry.json"
    update_registry(registry_path, manifest, promote=promote)

    return ArtifactBundle(
        artifact_id=artifact_id,
        bundle_dir=bundle_dir,
        model_path=bundled_model_path,
        manifest_path=manifest_path,
        registry_path=registry_path,
    )


def resolve_model_artifact(
    artifact_id: str,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> tuple[str, Path]:
    manifest_path = Path(artifact_root) / artifact_id / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Model artifact manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    backend = str(manifest["backend"])
    model_filename = str(manifest["model_filename"])
    return backend, manifest_path.parent / model_filename


def promote_artifact(
    artifact_id: str,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    gates: PromotionGateConfig | None = None,
) -> PromotionResult:
    selected_gates = gates or PromotionGateConfig()
    root = Path(artifact_root)
    manifest_path = root / artifact_id / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Model artifact manifest does not exist: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = promotion_gate_failures(
        manifest=manifest,
        bundle_dir=manifest_path.parent,
        gates=selected_gates,
    )
    if failures:
        return PromotionResult(artifact_id=artifact_id, promoted=False, failures=failures)

    manifest["status"] = "promoted"
    _write_json(manifest_path, manifest)
    model_path = manifest_path.parent / str(manifest["model_filename"])
    _write_json(model_manifest_path(model_path), manifest)
    update_registry(root / "registry.json", manifest, promote=True)
    return PromotionResult(artifact_id=artifact_id, promoted=True, failures=[])


def promotion_gate_failures(
    manifest: dict[str, Any],
    bundle_dir: Path,
    gates: PromotionGateConfig,
) -> list[str]:
    failures: list[str] = []
    required_keys = {
        "artifact_id",
        "backend",
        "created_at",
        "git_sha",
        "model_filename",
        "feature_schema_hash",
        "feature_schema_version",
        "source_data_fingerprint",
        "training_config",
        "evaluation_metrics",
        "evaluation_report",
    }
    if gates.require_replay_evaluation:
        required_keys.update(
            {
                "replay_evaluation_metrics",
                "replay_evaluation_report",
                "replay_dataset_fingerprint",
                "replay_dataset_path",
            }
        )
        if gates.require_production_replay_validation:
            required_keys.add("replay_data_provenance")
    if gates.require_replay_suite:
        required_keys.update(
            {
                "replay_suite_metrics",
                "replay_suite_report",
            }
        )
    missing = sorted(key for key in required_keys if key not in manifest)
    if missing:
        failures.append(f"manifest missing required keys: {', '.join(missing)}")

    if manifest.get("feature_schema_hash") != feature_schema_hash():
        failures.append(
            "feature schema hash mismatch: "
            f"artifact={manifest.get('feature_schema_hash')} expected={feature_schema_hash()}"
        )

    model_filename = manifest.get("model_filename")
    if not model_filename or not (bundle_dir / str(model_filename)).exists():
        failures.append(f"model file is missing: {model_filename}")

    evaluation_path = bundle_dir / str(manifest.get("evaluation_report", "evaluation.json"))
    if not evaluation_path.exists():
        failures.append(f"evaluation report is missing: {evaluation_path.name}")
        evaluation_payload: dict[str, Any] = {}
    else:
        evaluation_payload = json.loads(evaluation_path.read_text(encoding="utf-8"))

    metrics = dict(manifest.get("evaluation_metrics", {}))
    mean_mae = _float_metric(metrics, "mean_mae_lap_delta_s")
    if mean_mae is None or mean_mae > gates.max_mean_mae_lap_delta_s:
        failures.append(
            "mean MAE gate failed: "
            f"value={mean_mae} threshold<={gates.max_mean_mae_lap_delta_s}"
        )

    mean_coverage = _float_metric(metrics, "mean_coverage_pct")
    if mean_coverage is None or mean_coverage < gates.min_mean_coverage_pct:
        failures.append(
            "mean coverage gate failed: "
            f"value={mean_coverage} threshold>={gates.min_mean_coverage_pct}"
        )

    max_calibration_error = _float_metric(metrics, "max_calibration_error_pct")
    if (
        max_calibration_error is None
        or max_calibration_error > gates.max_calibration_error_pct
    ):
        failures.append(
            "calibration gate failed: "
            f"value={max_calibration_error} threshold<={gates.max_calibration_error_pct}"
        )

    max_interval_width = _float_metric(metrics, "max_mean_interval_width_s")
    if max_interval_width is None or max_interval_width > gates.max_mean_interval_width_s:
        failures.append(
            "interval sharpness gate failed: "
            f"value={max_interval_width} threshold<={gates.max_mean_interval_width_s}"
        )

    max_pit_error = _float_metric(metrics, "max_pit_target_error_laps")
    if max_pit_error is None or max_pit_error > gates.max_pit_target_error_laps:
        failures.append(
            "pit decision gate failed: "
            f"value={max_pit_error} threshold<={gates.max_pit_target_error_laps}"
        )

    max_strategy_regret = _float_metric(metrics, "max_strategy_regret_s")
    if max_strategy_regret is None or max_strategy_regret > gates.max_strategy_regret_s:
        failures.append(
            "strategy regret gate failed: "
            f"value={max_strategy_regret} threshold<={gates.max_strategy_regret_s}"
        )

    scenario_failures = _scenario_gate_failures(evaluation_payload, gates)
    failures.extend(scenario_failures)
    failures.extend(_replay_gate_failures(manifest, bundle_dir, gates))
    failures.extend(_replay_suite_gate_failures(manifest, bundle_dir, gates))
    return failures


def update_registry(registry_path: str | Path, manifest: dict[str, Any], promote: bool) -> None:
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
    else:
        registry = {"artifacts": [], "promoted": {}}

    artifact_entry = {
        "artifact_id": manifest["artifact_id"],
        "backend": manifest["backend"],
        "created_at": manifest["created_at"],
        "git_sha": manifest["git_sha"],
        "model_path": manifest["model_path"],
        "status": "promoted" if promote else "candidate",
        "mean_mae_lap_delta_s": manifest["evaluation_metrics"]["mean_mae_lap_delta_s"],
        "mean_coverage_pct": manifest["evaluation_metrics"]["mean_coverage_pct"],
        "replay_passed": manifest.get("replay_evaluation_metrics", {}).get("passed"),
        "replay_suite_passed": manifest.get("replay_suite_metrics", {}).get("passed"),
        "replay_suite_split_count": manifest.get("replay_suite_metrics", {}).get("split_count"),
        "replay_mae_lap_delta_s": manifest.get("replay_evaluation_metrics", {}).get(
            "mae_lap_delta_s"
        ),
        "replay_coverage_pct": manifest.get("replay_evaluation_metrics", {}).get("coverage_pct"),
        "replay_dataset_fingerprint": manifest.get("replay_dataset_fingerprint"),
        "replay_validation_signal": manifest.get("replay_evaluation_metrics", {}).get(
            "validation_signal"
        ),
        "production_validation_ready": manifest.get("replay_evaluation_metrics", {}).get(
            "production_validation_ready",
            False,
        ),
        "feature_schema_hash": manifest["feature_schema_hash"],
    }

    artifacts = [
        item
        for item in registry.get("artifacts", [])
        if item.get("artifact_id") != artifact_entry["artifact_id"]
    ]
    artifacts.append(artifact_entry)
    artifacts.sort(key=lambda item: item["created_at"], reverse=True)
    registry["artifacts"] = artifacts
    promoted = dict(registry.get("promoted", {}))
    if promote:
        promoted[str(manifest["backend"])] = manifest["artifact_id"]
    registry["promoted"] = promoted
    _write_json(path, registry)


def training_data_fingerprint(training_config: dict[str, Any]) -> str:
    payload = {
        "source": "deterministic-race-simulation",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": feature_schema_hash(),
        "training_config": training_config,
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def current_git_sha() -> str:
    env_sha = os.getenv("F1_BUILD_SHA")
    if env_sha and env_sha != "unknown":
        return env_sha[:12]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage local model artifact registry.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_parser = subparsers.add_parser(
        "register",
        help="Register an existing local model file as a versioned artifact bundle.",
    )
    register_parser.add_argument("--backend", required=True, choices=sorted(LOCAL_MODEL_PATHS))
    register_parser.add_argument("--model-path", required=True)
    register_parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    register_parser.add_argument("--replay-dataset", default="examples/replay_telemetry.csv")
    register_parser.add_argument("--promote", action="store_true")
    register_local_parser = subparsers.add_parser(
        "register-local",
        help="Register all known model files under models/ as versioned artifact bundles.",
    )
    register_local_parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    register_local_parser.add_argument("--replay-dataset", default="examples/replay_telemetry.csv")
    register_local_parser.add_argument("--promote", action="store_true")
    prune_parser = subparsers.add_parser(
        "prune",
        help="Archive older candidate artifacts while keeping promoted artifacts.",
    )
    prune_parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    prune_parser.add_argument("--keep-candidates-per-backend", type=int, default=2)
    promote_parser = subparsers.add_parser("promote", help="Promote a validated artifact.")
    promote_parser.add_argument("--artifact-id", required=True)
    promote_parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    promote_parser.add_argument("--max-mean-mae", type=float, default=1.25)
    promote_parser.add_argument("--min-mean-coverage", type=float, default=50.0)
    promote_parser.add_argument("--max-latency-p95-ms", type=float, default=25.0)
    promote_parser.add_argument("--max-wear-violations", type=int, default=0)
    promote_parser.add_argument("--max-calibration-error-pct", type=float, default=20.0)
    promote_parser.add_argument("--max-mean-interval-width", type=float, default=5.5)
    promote_parser.add_argument("--max-replay-mae", type=float, default=0.35)
    promote_parser.add_argument("--min-replay-coverage", type=float, default=80.0)
    promote_parser.add_argument("--max-replay-missing-target-pct", type=float, default=0.0)
    promote_parser.add_argument("--min-replay-event-count", type=int, default=12)
    promote_parser.add_argument("--max-pit-target-error-laps", type=float, default=7.0)
    promote_parser.add_argument("--max-strategy-regret", type=float, default=2.5)
    promote_parser.add_argument("--required-replay-suite-name", default="benchmark")
    promote_parser.add_argument("--min-replay-suite-split-count", type=int, default=5)
    promote_parser.add_argument(
        "--require-production-replay-validation",
        action="store_true",
        help="Require replay manifests with observed public labels rather than proxy-heavy data.",
    )
    promote_latest_parser = subparsers.add_parser(
        "promote-latest",
        help="Promote the most recent candidate artifact for a backend (used in CI pipelines).",
    )
    promote_latest_parser.add_argument("--backend", required=True, choices=sorted(LOCAL_MODEL_PATHS))
    promote_latest_parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    rollback_parser = subparsers.add_parser(
        "rollback-candidate",
        help="Select the latest promoted rollback artifact for a backend.",
    )
    rollback_parser.add_argument("--backend", required=True)
    rollback_parser.add_argument("--active-artifact-id", default="")
    rollback_parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "register":
        bundle = register_existing_model_artifact(
            model_path=args.model_path,
            backend=args.backend,
            artifact_root=args.artifact_root,
            replay_dataset=args.replay_dataset,
            promote=args.promote,
        )
        print(f"registered artifact: {bundle.artifact_id}")
        print(f"bundle: {bundle.bundle_dir}")
        print(f"registry: {bundle.registry_path}")
        return
    if args.command == "register-local":
        results = register_local_model_artifacts(
            artifact_root=args.artifact_root,
            replay_dataset=args.replay_dataset,
            promote=args.promote,
        )
        failed = False
        for result in results:
            if result.bundle is None:
                failed = True
                print(f"skipped {result.backend}: {result.skipped_reason}")
            else:
                print(f"registered {result.backend}: {result.bundle.artifact_id}")
        if failed:
            raise SystemExit(1)
        return
    if args.command == "prune":
        result = prune_artifact_registry(
            artifact_root=args.artifact_root,
            keep_candidates_per_backend=args.keep_candidates_per_backend,
        )
        print(f"kept artifacts: {len(result.kept)}")
        print(f"archived artifacts: {len(result.archived)}")
        for artifact_id in result.archived:
            print(f"- {artifact_id}")
        return
    if args.command == "promote":
        result = promote_artifact(
            artifact_id=args.artifact_id,
            artifact_root=args.artifact_root,
            gates=PromotionGateConfig(
                max_mean_mae_lap_delta_s=args.max_mean_mae,
                min_mean_coverage_pct=args.min_mean_coverage,
                max_calibration_error_pct=args.max_calibration_error_pct,
                max_mean_interval_width_s=args.max_mean_interval_width,
                max_latency_p95_ms=args.max_latency_p95_ms,
                max_monotonic_wear_violations=args.max_wear_violations,
                max_replay_mae_lap_delta_s=args.max_replay_mae,
                min_replay_coverage_pct=args.min_replay_coverage,
                max_replay_missing_target_pct=args.max_replay_missing_target_pct,
                min_replay_event_count=args.min_replay_event_count,
                max_pit_target_error_laps=args.max_pit_target_error_laps,
                max_strategy_regret_s=args.max_strategy_regret,
                required_replay_suite_name=args.required_replay_suite_name,
                min_replay_suite_split_count=args.min_replay_suite_split_count,
                require_production_replay_validation=args.require_production_replay_validation,
            ),
        )
        if result.promoted:
            print(f"promoted artifact: {result.artifact_id}")
            return
        print(f"artifact promotion failed: {result.artifact_id}")
        for failure in result.failures:
            print(f"- {failure}")
        raise SystemExit(1)
    if args.command == "promote-latest":
        registry = load_registry(args.artifact_root)
        candidates = [
            r for r in registry.get("artifacts", [])
            if r.get("backend") == args.backend and r.get("status") == "candidate"
        ]
        if not candidates:
            # All artifacts are already promoted — nothing to do (idempotent for DVC).
            print(f"no unreviewed candidates for backend={args.backend} — already up to date")
            return
        # Sort by the ISO timestamp embedded in artifact_id (format: backend/TIMESTAMP-SHA).
        # Newest artifact has the lexicographically largest timestamp.
        candidates.sort(key=lambda r: r["artifact_id"].split("/", 1)[-1], reverse=True)
        latest = candidates[0]["artifact_id"]
        result = promote_artifact(artifact_id=latest, artifact_root=args.artifact_root)
        if result.promoted:
            print(f"promoted artifact: {result.artifact_id}")
            return
        print(f"artifact promotion failed: {result.artifact_id}")
        for failure in result.failures:
            print(f"- {failure}")
        raise SystemExit(1)
    if args.command == "rollback-candidate":
        candidate = rollback_candidate(
            load_registry(args.artifact_root),
            backend=args.backend,
            active_artifact_id=args.active_artifact_id or "",
        )
        if candidate is None:
            print("no rollback candidate")
            raise SystemExit(1)
        print(json.dumps(candidate, indent=2, sort_keys=True))
        return
    raise ValueError(f"Unsupported artifact command: {args.command}")


def _artifact_manifest(
    artifact_id: str,
    backend: str,
    created_at: str,
    git_sha: str,
    model_filename: str,
    training_config: dict[str, Any],
    evaluation_payload: dict[str, Any],
    replay_payload: dict[str, Any] | None,
    replay_suite_payload: dict[str, Any] | None,
    promoted: bool,
) -> dict[str, Any]:
    manifest = model_manifest(
        backend=backend,
        training_rows=int(training_config.get("training_rows") or 0),
        feature_names=FEATURE_NAMES,
    )
    manifest.update(
        {
            "artifact_id": artifact_id,
            "created_at": created_at,
            "git_sha": git_sha,
            "app_version": APP_VERSION,
            "model_filename": model_filename,
            "model_path": f"{artifact_id}/{model_filename}",
            "training_config": training_config,
            "training_parameters": training_config,
            "source_data_fingerprint": training_data_fingerprint(training_config),
            "evaluation_metrics": {
                "scenario_count": evaluation_payload["scenario_count"],
                "mean_mae_lap_delta_s": evaluation_payload["mean_mae_lap_delta_s"],
                "mean_coverage_pct": evaluation_payload["mean_coverage_pct"],
                "max_calibration_error_pct": max(
                    float(item.get("calibration_error_pct", 0.0))
                    for item in evaluation_payload["scenarios"]
                ),
                "max_mean_interval_width_s": max(
                    float(item.get("mean_interval_width_s", 0.0))
                    for item in evaluation_payload["scenarios"]
                ),
                "max_pit_target_error_laps": max(
                    float(item.get("pit_target_error_laps", 0.0))
                    for item in evaluation_payload["scenarios"]
                ),
                "max_strategy_regret_s": max(
                    float(item.get("strategy_regret_s", 0.0))
                    for item in evaluation_payload["scenarios"]
                ),
            },
            "evaluation_report": "evaluation.json",
            "status": "promoted" if promoted else "candidate",
        }
    )
    if replay_payload is not None:
        scenario = dict(replay_payload.get("scenario", {}))
        manifest.update(
            {
                "replay_evaluation_report": "replay_evaluation.json",
                "replay_dataset_path": replay_payload["dataset_path"],
                "replay_dataset_fingerprint": replay_payload["dataset_fingerprint"],
                "replay_data_provenance": replay_payload.get("data_provenance", {}),
                "replay_evaluation_metrics": {
                    "passed": replay_payload["passed"],
                    "event_count": replay_payload["event_count"],
                    "labeled_event_count": replay_payload["labeled_event_count"],
                    "missing_target_pct": replay_payload["missing_target_pct"],
                    "mae_lap_delta_s": scenario.get("mae_lap_delta_s"),
                    "rmse_lap_delta_s": scenario.get("rmse_lap_delta_s"),
                    "coverage_pct": scenario.get("coverage_pct"),
                    "latency_p95_ms": scenario.get("latency_p95_ms"),
                    "monotonic_wear_violations": scenario.get("monotonic_wear_violations"),
                    "mean_interval_width_s": scenario.get("mean_interval_width_s"),
                    "calibration_error_pct": scenario.get("calibration_error_pct"),
                    "pit_target_error_laps": scenario.get("pit_target_error_laps"),
                    "strategy_regret_s": scenario.get("strategy_regret_s"),
                    "gates": replay_payload["gates"],
                    "validation_signal": replay_payload.get("data_provenance", {}).get(
                        "validation_signal"
                    ),
                    "production_validation_ready": replay_payload.get(
                        "data_provenance", {}
                    ).get("production_validation_ready", False),
                },
            }
        )
    if replay_suite_payload is not None:
        manifest.update(
            {
                "replay_suite_report": "replay_suite.json",
                "replay_suite_metrics": {
                    "suite_name": replay_suite_payload.get("suite_name", "default"),
                    "passed": replay_suite_payload["passed"],
                    "split_count": replay_suite_payload["split_count"],
                    "mean_mae_lap_delta_s": replay_suite_payload["mean_mae_lap_delta_s"],
                    "mean_coverage_pct": replay_suite_payload["mean_coverage_pct"],
                    "total_event_count": replay_suite_payload["total_event_count"],
                    "total_labeled_event_count": replay_suite_payload[
                        "total_labeled_event_count"
                    ],
                    "splits": [
                        {
                            "name": split["scenario"]["scenario"],
                            "passed": split["passed"],
                            "dataset_fingerprint": split["dataset_fingerprint"],
                            "mae_lap_delta_s": split["scenario"]["mae_lap_delta_s"],
                            "coverage_pct": split["scenario"]["coverage_pct"],
                            "latency_p95_ms": split["scenario"]["latency_p95_ms"],
                            "missing_target_pct": split["missing_target_pct"],
                            "event_count": split["event_count"],
                            "mean_interval_width_s": split["scenario"][
                                "mean_interval_width_s"
                            ],
                            "calibration_error_pct": split["scenario"][
                                "calibration_error_pct"
                            ],
                            "pit_target_error_laps": split["scenario"][
                                "pit_target_error_laps"
                            ],
                            "strategy_regret_s": split["scenario"]["strategy_regret_s"],
                            "validation_signal": split.get("data_provenance", {}).get(
                                "validation_signal"
                            ),
                            "production_validation_ready": split.get(
                                "data_provenance", {}
                            ).get("production_validation_ready", False),
                        }
                        for split in replay_suite_payload["splits"]
                    ],
                },
            }
        )
    return manifest


def _model_card_payload(
    *,
    manifest: dict[str, Any],
    evaluation_payload: dict[str, Any],
    replay_payload: dict[str, Any] | None,
    replay_suite_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    replay_provenance = dict((replay_payload or {}).get("data_provenance", {}))
    replay_metrics = dict(manifest.get("replay_evaluation_metrics", {}))
    suite_metrics = dict(manifest.get("replay_suite_metrics", {}))
    return {
        "artifact_id": manifest["artifact_id"],
        "backend": manifest["backend"],
        "app_version": manifest.get("app_version"),
        "created_at": manifest["created_at"],
        "git_sha": manifest["git_sha"],
        "feature_schema_version": manifest["feature_schema_version"],
        "feature_schema_hash": manifest["feature_schema_hash"],
        "training": {
            "source": "deterministic-race-simulation",
            "parameters": manifest["training_config"],
            "source_data_fingerprint": manifest["source_data_fingerprint"],
        },
        "simulator_evaluation": manifest["evaluation_metrics"],
        "replay_evaluation": {
            "dataset_path": manifest.get("replay_dataset_path"),
            "dataset_fingerprint": manifest.get("replay_dataset_fingerprint"),
            "metrics": replay_metrics,
            "provenance": replay_provenance,
        },
        "benchmark_suite": suite_metrics,
        "promotion": {
            "status": manifest["status"],
            "production_validation_ready": replay_metrics.get(
                "production_validation_ready",
                False,
            ),
            "validation_signal": replay_metrics.get("validation_signal"),
            "replay_passed": replay_metrics.get("passed"),
            "benchmark_passed": suite_metrics.get("passed"),
        },
        "limitations": replay_provenance.get("limitations", []),
        "reports": {
            "evaluation": manifest.get("evaluation_report"),
            "replay_evaluation": manifest.get("replay_evaluation_report"),
            "replay_suite": manifest.get("replay_suite_report"),
        },
        "summary": {
            "mean_mae_lap_delta_s": evaluation_payload.get("mean_mae_lap_delta_s"),
            "mean_coverage_pct": evaluation_payload.get("mean_coverage_pct"),
            "replay_mae_lap_delta_s": replay_metrics.get("mae_lap_delta_s"),
            "replay_coverage_pct": replay_metrics.get("coverage_pct"),
            "replay_mean_interval_width_s": replay_metrics.get("mean_interval_width_s"),
            "observed_field_count": replay_provenance.get("observed_field_count"),
            "proxy_diagnostic_field_count": replay_provenance.get(
                "proxy_diagnostic_field_count"
            ),
        },
    }


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.name or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _training_rows_from_manifest(model_path: Path) -> int | None:
    path = model_manifest_path(model_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("training_rows")
        return int(value) if value is not None else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _float_metric(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _scenario_gate_failures(
    evaluation_payload: dict[str, Any],
    gates: PromotionGateConfig,
) -> list[str]:
    failures: list[str] = []
    scenarios = evaluation_payload.get("scenarios", [])
    if not scenarios:
        failures.append("evaluation report has no scenarios")
        return failures
    for scenario in scenarios:
        name = scenario.get("scenario", "unknown")
        latency = _float_metric(scenario, "latency_p95_ms")
        if latency is None or latency > gates.max_latency_p95_ms:
            failures.append(
                f"{name} latency gate failed: "
                f"value={latency} threshold<={gates.max_latency_p95_ms}"
            )
        violations = scenario.get("monotonic_wear_violations")
        try:
            violation_count = int(violations)
        except (TypeError, ValueError):
            violation_count = gates.max_monotonic_wear_violations + 1
        if violation_count > gates.max_monotonic_wear_violations:
            failures.append(
                f"{name} monotonic wear gate failed: "
                f"value={violation_count} threshold<={gates.max_monotonic_wear_violations}"
            )
        calibration_error = _float_metric(scenario, "calibration_error_pct")
        if (
            calibration_error is None
            or calibration_error > gates.max_calibration_error_pct
        ):
            failures.append(
                f"{name} calibration gate failed: "
                f"value={calibration_error} threshold<={gates.max_calibration_error_pct}"
            )
        interval_width = _float_metric(scenario, "mean_interval_width_s")
        if interval_width is None or interval_width > gates.max_mean_interval_width_s:
            failures.append(
                f"{name} interval sharpness gate failed: "
                f"value={interval_width} threshold<={gates.max_mean_interval_width_s}"
            )
        pit_error = _float_metric(scenario, "pit_target_error_laps")
        if pit_error is None or pit_error > gates.max_pit_target_error_laps:
            failures.append(
                f"{name} pit decision gate failed: "
                f"value={pit_error} threshold<={gates.max_pit_target_error_laps}"
            )
        strategy_regret = _float_metric(scenario, "strategy_regret_s")
        if strategy_regret is None or strategy_regret > gates.max_strategy_regret_s:
            failures.append(
                f"{name} strategy regret gate failed: "
                f"value={strategy_regret} threshold<={gates.max_strategy_regret_s}"
            )
    return failures


def is_registry_promoted(registry: dict[str, Any], artifact_id: str) -> bool:
    return any(
        artifact.get("artifact_id") == artifact_id and artifact.get("status") == "promoted"
        for artifact in registry.get("artifacts", [])
    )


def _replay_gate_failures(
    manifest: dict[str, Any],
    bundle_dir: Path,
    gates: PromotionGateConfig,
) -> list[str]:
    failures: list[str] = []
    if not gates.require_replay_evaluation:
        return failures
    report_name = manifest.get("replay_evaluation_report")
    if not report_name:
        failures.append("replay evaluation report is missing from manifest")
        return failures

    replay_path = bundle_dir / str(report_name)
    if not replay_path.exists():
        failures.append(f"replay evaluation report is missing: {replay_path.name}")
        replay_payload: dict[str, Any] = {}
    else:
        replay_payload = json.loads(replay_path.read_text(encoding="utf-8"))

    metrics = dict(manifest.get("replay_evaluation_metrics", {}))

    if gates.require_production_replay_validation:
        production_ready = metrics.get("production_validation_ready") is True
        if not production_ready:
            failures.append(
                "replay production validation gate failed: "
                f"validation_signal={metrics.get('validation_signal')}"
            )

    replay_mae = _float_metric(metrics, "mae_lap_delta_s")
    if replay_mae is None or replay_mae > gates.max_real_replay_mae_lap_delta_s:
        failures.append(
            "replay MAE gate failed: "
            f"value={replay_mae} threshold<={gates.max_real_replay_mae_lap_delta_s}"
        )

    replay_coverage = _float_metric(metrics, "coverage_pct")
    if replay_coverage is None or replay_coverage < gates.min_replay_coverage_pct:
        failures.append(
            "replay coverage gate failed: "
            f"value={replay_coverage} threshold>={gates.min_replay_coverage_pct}"
        )

    missing_targets = _float_metric(metrics, "missing_target_pct")
    if missing_targets is None or missing_targets > gates.max_replay_missing_target_pct:
        failures.append(
            "replay target completeness gate failed: "
            f"value={missing_targets} threshold<={gates.max_replay_missing_target_pct}"
        )

    try:
        event_count = int(metrics.get("event_count"))
    except (TypeError, ValueError):
        event_count = 0
    if event_count < gates.min_replay_event_count:
        failures.append(
            "replay sample-size gate failed: "
            f"value={event_count} threshold>={gates.min_replay_event_count}"
        )

    interval_width = _float_metric(metrics, "mean_interval_width_s")
    if interval_width is None or interval_width > gates.max_mean_interval_width_s:
        failures.append(
            "replay interval sharpness gate failed: "
            f"value={interval_width} threshold<={gates.max_mean_interval_width_s}"
        )

    calibration_error = _float_metric(metrics, "calibration_error_pct")
    if (
        calibration_error is None
        or calibration_error > gates.max_calibration_error_pct
    ):
        failures.append(
            "replay calibration gate failed: "
            f"value={calibration_error} threshold<={gates.max_calibration_error_pct}"
        )

    pit_error = _float_metric(metrics, "pit_target_error_laps")
    if pit_error is None or pit_error > gates.max_pit_target_error_laps:
        failures.append(
            "replay pit decision gate failed: "
            f"value={pit_error} threshold<={gates.max_pit_target_error_laps}"
        )

    strategy_regret = _float_metric(metrics, "strategy_regret_s")
    if strategy_regret is None or strategy_regret > gates.max_strategy_regret_s:
        failures.append(
            "replay strategy regret gate failed: "
            f"value={strategy_regret} threshold<={gates.max_strategy_regret_s}"
        )

    latency = _float_metric(metrics, "latency_p95_ms")
    if latency is None or latency > gates.max_latency_p95_ms:
        failures.append(
            "replay latency gate failed: "
            f"value={latency} threshold<={gates.max_latency_p95_ms}"
        )

    try:
        wear_violations = int(metrics.get("monotonic_wear_violations"))
    except (TypeError, ValueError):
        wear_violations = gates.max_real_replay_monotonic_wear_violations + 1
    if wear_violations > gates.max_real_replay_monotonic_wear_violations:
        failures.append(
            "replay monotonic wear gate failed: "
            f"value={wear_violations} threshold<={gates.max_real_replay_monotonic_wear_violations}"
        )

    manifest_fingerprint = manifest.get("replay_dataset_fingerprint")
    report_fingerprint = replay_payload.get("dataset_fingerprint")
    if manifest_fingerprint != report_fingerprint:
        failures.append(
            "replay dataset fingerprint mismatch: "
            f"manifest={manifest_fingerprint} report={report_fingerprint}"
        )

    return failures


def _replay_suite_gate_failures(
    manifest: dict[str, Any],
    bundle_dir: Path,
    gates: PromotionGateConfig,
) -> list[str]:
    failures: list[str] = []
    report_name = manifest.get("replay_suite_report")
    if not report_name:
        if gates.require_replay_suite:
            failures.append("replay suite report is missing from manifest")
        return failures

    suite_path = bundle_dir / str(report_name)
    if not suite_path.exists():
        failures.append(f"replay suite report is missing: {suite_path.name}")
        suite_payload: dict[str, Any] = {}
    else:
        suite_payload = json.loads(suite_path.read_text(encoding="utf-8"))

    metrics = dict(manifest.get("replay_suite_metrics", {}))
    if metrics.get("passed") is not True:
        failures.append(f"replay suite summary failed: passed={metrics.get('passed')}")

    if gates.required_replay_suite_name:
        suite_name = str(metrics.get("suite_name", ""))
        if suite_name != gates.required_replay_suite_name:
            failures.append(
                "replay suite name gate failed: "
                f"value={suite_name} expected={gates.required_replay_suite_name}"
            )

    try:
        split_count = int(metrics.get("split_count"))
    except (TypeError, ValueError):
        split_count = 0
    if split_count < gates.min_replay_suite_split_count:
        failures.append(
            "replay suite split-count gate failed: "
            f"value={split_count} threshold>={gates.min_replay_suite_split_count}"
        )

    split_metrics = metrics.get("splits", [])
    if not split_metrics:
        failures.append("replay suite has no split metrics")

    for split in split_metrics:
        name = split.get("name", "unknown")
        if split.get("passed") is not True:
            failures.append(f"{name} replay split failed")
        mae = _float_metric(split, "mae_lap_delta_s")
        if mae is None or mae > gates.max_replay_mae_lap_delta_s:
            failures.append(
                f"{name} replay MAE gate failed: "
                f"value={mae} threshold<={gates.max_replay_mae_lap_delta_s}"
            )
        coverage = _float_metric(split, "coverage_pct")
        if coverage is None or coverage < gates.min_replay_coverage_pct:
            failures.append(
                f"{name} replay coverage gate failed: "
                f"value={coverage} threshold>={gates.min_replay_coverage_pct}"
            )
        latency = _float_metric(split, "latency_p95_ms")
        if latency is None or latency > gates.max_latency_p95_ms:
            failures.append(
                f"{name} replay latency gate failed: "
                f"value={latency} threshold<={gates.max_latency_p95_ms}"
            )
        missing_targets = _float_metric(split, "missing_target_pct")
        if missing_targets is None or missing_targets > gates.max_replay_missing_target_pct:
            failures.append(
                f"{name} replay target completeness gate failed: "
                f"value={missing_targets} threshold<={gates.max_replay_missing_target_pct}"
            )
        try:
            event_count = int(split.get("event_count"))
        except (TypeError, ValueError):
            event_count = 0
        if event_count < gates.min_replay_event_count:
            failures.append(
                f"{name} replay sample-size gate failed: "
                f"value={event_count} threshold>={gates.min_replay_event_count}"
            )
        interval_width = _float_metric(split, "mean_interval_width_s")
        if interval_width is None or interval_width > gates.max_mean_interval_width_s:
            failures.append(
                f"{name} replay interval sharpness gate failed: "
                f"value={interval_width} threshold<={gates.max_mean_interval_width_s}"
            )
        calibration_error = _float_metric(split, "calibration_error_pct")
        if (
            calibration_error is None
            or calibration_error > gates.max_calibration_error_pct
        ):
            failures.append(
                f"{name} replay calibration gate failed: "
                f"value={calibration_error} threshold<={gates.max_calibration_error_pct}"
            )
        pit_error = _float_metric(split, "pit_target_error_laps")
        if pit_error is None or pit_error > gates.max_pit_target_error_laps:
            failures.append(
                f"{name} replay pit decision gate failed: "
                f"value={pit_error} threshold<={gates.max_pit_target_error_laps}"
            )
        strategy_regret = _float_metric(split, "strategy_regret_s")
        if strategy_regret is None or strategy_regret > gates.max_strategy_regret_s:
            failures.append(
                f"{name} replay strategy regret gate failed: "
                f"value={strategy_regret} threshold<={gates.max_strategy_regret_s}"
            )

    report_splits = suite_payload.get("splits", [])
    report_failures = [
        split.get("scenario", {}).get("scenario", "unknown")
        for split in report_splits
        if split.get("passed") is not True
    ]
    if report_failures:
        failures.append(f"replay suite report splits failed: {', '.join(report_failures)}")

    return failures


if __name__ == "__main__":
    main()
