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


DEFAULT_ARTIFACT_ROOT = "artifacts/models"


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
    max_latency_p95_ms: float = 25.0
    max_monotonic_wear_violations: int = 0


@dataclass(frozen=True)
class PromotionResult:
    artifact_id: str
    promoted: bool
    failures: list[str]


def create_model_artifact_bundle(
    model_path: str | Path,
    backend: str,
    training_config: dict[str, Any],
    evaluation_report: EvaluationReport,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
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
    manifest = _artifact_manifest(
        artifact_id=artifact_id,
        backend=backend,
        created_at=timestamp,
        git_sha=resolved_git_sha,
        model_filename=bundled_model_path.name,
        training_config=training_config,
        evaluation_payload=evaluation_payload,
        promoted=promote,
    )

    manifest_path = bundle_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(model_manifest_path(bundled_model_path), manifest)
    _write_json(bundle_dir / "training_config.json", training_config)
    _write_json(bundle_dir / "evaluation.json", evaluation_payload)
    (bundle_dir / "evaluation.md").write_text(render_markdown(evaluation_report), encoding="utf-8")

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

    scenario_failures = _scenario_gate_failures(evaluation_payload, gates)
    failures.extend(scenario_failures)
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
    promote_parser = subparsers.add_parser("promote", help="Promote a validated artifact.")
    promote_parser.add_argument("--artifact-id", required=True)
    promote_parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    promote_parser.add_argument("--max-mean-mae", type=float, default=1.25)
    promote_parser.add_argument("--min-mean-coverage", type=float, default=50.0)
    promote_parser.add_argument("--max-latency-p95-ms", type=float, default=25.0)
    promote_parser.add_argument("--max-wear-violations", type=int, default=0)
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
    if args.command == "promote":
        result = promote_artifact(
            artifact_id=args.artifact_id,
            artifact_root=args.artifact_root,
            gates=PromotionGateConfig(
                max_mean_mae_lap_delta_s=args.max_mean_mae,
                min_mean_coverage_pct=args.min_mean_coverage,
                max_latency_p95_ms=args.max_latency_p95_ms,
                max_monotonic_wear_violations=args.max_wear_violations,
            ),
        )
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
    promoted: bool,
) -> dict[str, Any]:
    manifest = model_manifest(
        backend=backend,
        training_rows=int(training_config.get("training_rows", 0)),
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
            },
            "evaluation_report": "evaluation.json",
            "status": "promoted" if promoted else "candidate",
        }
    )
    return manifest


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    return failures


if __name__ == "__main__":
    main()
