from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from dataclasses import field, replace as _replace
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

from f1_strategy.domain import TelemetryEvent, TireCompound
from f1_strategy.engine import InferenceEngine
from f1_strategy.evaluation import ScenarioEvaluation, _oracle_pit_lap, _strategy_regret_s
from f1_strategy.metadata import APP_VERSION
from f1_strategy.models import FEATURE_SCHEMA_VERSION, feature_schema_hash
from f1_strategy.serialization import telemetry_from_dict, to_jsonable
from f1_strategy.simulation import RaceSimulator, SimulationConfig

REPLAY_REQUIRED_COLUMNS = [
    "session_id",
    "car_id",
    "lap",
    "sector",
    "speed_kph",
    "throttle",
    "brake",
    "steering_angle",
    "tire_temp_fl",
    "tire_temp_fr",
    "tire_temp_rl",
    "tire_temp_rr",
    "brake_temp",
    "slip_angle",
    "lateral_g",
    "ers_soc",
    "ers_deployment_kw",
    "fuel_kg",
    "track_temp_c",
    "air_temp_c",
    "humidity",
    "compound",
]

REPLAY_OPTIONAL_COLUMNS = ["lap_time_s", "timestamp_ms", "circuit"]
REPLAY_IGNORED_COLUMNS = ["actual_tire_age_laps"]
DEFAULT_REPLAY_DATASET = Path("examples/replay_telemetry.csv")


@dataclass(frozen=True)
class ReplayGateConfig:
    max_mae_lap_delta_s: float = 0.35
    min_coverage_pct: float = 80.0
    max_calibration_error_pct: float = 20.0
    max_mean_interval_width_s: float = 5.5
    max_latency_p95_ms: float = 25.0
    max_monotonic_wear_violations: int = 0
    max_missing_target_pct: float = 0.0
    min_event_count: int = 12
    max_pit_target_error_laps: float = 7.0
    max_strategy_regret_s: float = 2.5
    require_observed_lap_time_labels: bool = False


@dataclass(frozen=True)
class ReplaySplitSpec:
    name: str
    dataset_path: Path | None = None
    laps: int = 8
    seed: int = 101
    compound: TireCompound = TireCompound.MEDIUM
    track_temp_offset_c: float = 0.0
    dirty_air: bool = False
    base_lap_time_s: float = 90.0
    gates: ReplayGateConfig | None = None


@dataclass(frozen=True)
class ReplayEvaluationReport:
    version: str
    feature_schema_version: str
    feature_schema_hash: str
    dataset_path: str
    dataset_fingerprint: str
    session_count: int
    event_count: int
    labeled_event_count: int
    missing_target_pct: float
    scenario: ScenarioEvaluation
    gates: dict[str, bool]
    passed: bool
    data_provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplaySuiteReport:
    version: str
    feature_schema_version: str
    feature_schema_hash: str
    split_count: int
    passed: bool
    mean_mae_lap_delta_s: float
    mean_coverage_pct: float
    total_event_count: int
    total_labeled_event_count: int
    splits: list[ReplayEvaluationReport]
    suite_name: str = "default"


DEFAULT_REPLAY_SUITE = [
    ReplaySplitSpec("smoke", dataset_path=DEFAULT_REPLAY_DATASET),
    ReplaySplitSpec("holdout", laps=10, seed=201, compound=TireCompound.MEDIUM),
    ReplaySplitSpec(
        "hot-track",
        laps=10,
        seed=202,
        compound=TireCompound.SOFT,
        track_temp_offset_c=11.0,
    ),
    ReplaySplitSpec(
        "wet-intermediate",
        laps=8,
        seed=203,
        compound=TireCompound.INTERMEDIATE,
        base_lap_time_s=91.8,
    ),
    ReplaySplitSpec("long-run", laps=18, seed=204, compound=TireCompound.HARD),
    ReplaySplitSpec(
        "dirty-air",
        laps=10,
        seed=205,
        compound=TireCompound.MEDIUM,
        dirty_air=True,
    ),
]

BENCHMARK_REPLAY_DIR = Path("examples/replay_benchmarks")
BENCHMARK_MANIFEST_GENERATED_AT = "2026-06-05T00:00:00Z"
BENCHMARK_REPLAY_SUITE = [
    ReplaySplitSpec(
        "medium-long-run",
        dataset_path=BENCHMARK_REPLAY_DIR / "medium_long_run.csv",
        base_lap_time_s=90.0,
        gates=ReplayGateConfig(
            max_mae_lap_delta_s=0.25,
            min_coverage_pct=85.0,
            max_mean_interval_width_s=4.0,
            min_event_count=45,
            max_pit_target_error_laps=5.0,
            max_strategy_regret_s=2.0,
        ),
    ),
    ReplaySplitSpec(
        "soft-hot-track",
        dataset_path=BENCHMARK_REPLAY_DIR / "soft_hot_track.csv",
        base_lap_time_s=90.0,
        gates=ReplayGateConfig(
            max_mae_lap_delta_s=0.30,
            min_coverage_pct=85.0,
            max_mean_interval_width_s=4.5,
            min_event_count=36,
            max_pit_target_error_laps=5.0,
            max_strategy_regret_s=2.0,
        ),
    ),
    ReplaySplitSpec(
        "hard-fuel-burn",
        dataset_path=BENCHMARK_REPLAY_DIR / "hard_fuel_burn.csv",
        base_lap_time_s=90.0,
        gates=ReplayGateConfig(
            max_mae_lap_delta_s=0.25,
            min_coverage_pct=85.0,
            max_mean_interval_width_s=4.0,
            min_event_count=54,
            max_pit_target_error_laps=5.0,
            max_strategy_regret_s=2.0,
        ),
    ),
    ReplaySplitSpec(
        "intermediate-wet-track",
        dataset_path=BENCHMARK_REPLAY_DIR / "intermediate_wet_track.csv",
        base_lap_time_s=91.8,
        gates=ReplayGateConfig(
            max_mae_lap_delta_s=0.35,
            min_coverage_pct=80.0,
            max_mean_interval_width_s=5.0,
            min_event_count=36,
            max_pit_target_error_laps=6.0,
            max_strategy_regret_s=2.25,
        ),
    ),
    ReplaySplitSpec(
        "dirty-air-traffic",
        dataset_path=BENCHMARK_REPLAY_DIR / "dirty_air_traffic.csv",
        base_lap_time_s=90.0,
        gates=ReplayGateConfig(
            max_mae_lap_delta_s=0.30,
            min_coverage_pct=85.0,
            max_mean_interval_width_s=4.5,
            min_event_count=36,
            max_pit_target_error_laps=5.0,
            max_strategy_regret_s=2.0,
        ),
    ),
]

BENCHMARK_FIELD_PROVENANCE = {
    "session_id": "synthetic: deterministic benchmark fixture identifier",
    "car_id": "synthetic: deterministic benchmark fixture car identifier",
    "lap": "synthetic: deterministic benchmark fixture lap index",
    "sector": "synthetic: deterministic benchmark fixture sector index",
    "speed_kph": "synthetic: deterministic benchmark fixture telemetry",
    "throttle": "synthetic: deterministic benchmark fixture telemetry",
    "brake": "synthetic: deterministic benchmark fixture telemetry",
    "steering_angle": "synthetic: deterministic benchmark fixture telemetry",
    "tire_temp_fl": "synthetic: deterministic benchmark fixture telemetry",
    "tire_temp_fr": "synthetic: deterministic benchmark fixture telemetry",
    "tire_temp_rl": "synthetic: deterministic benchmark fixture telemetry",
    "tire_temp_rr": "synthetic: deterministic benchmark fixture telemetry",
    "brake_temp": "synthetic: deterministic benchmark fixture telemetry",
    "slip_angle": "synthetic: deterministic benchmark fixture telemetry",
    "lateral_g": "synthetic: deterministic benchmark fixture telemetry",
    "ers_soc": "synthetic: deterministic benchmark fixture telemetry",
    "ers_deployment_kw": "synthetic: deterministic benchmark fixture telemetry",
    "fuel_kg": "synthetic: deterministic benchmark fixture telemetry",
    "track_temp_c": "synthetic: deterministic benchmark fixture telemetry",
    "air_temp_c": "synthetic: deterministic benchmark fixture telemetry",
    "humidity": "synthetic: deterministic benchmark fixture telemetry",
    "compound": "synthetic: deterministic benchmark fixture compound",
    "lap_time_s": "synthetic: deterministic benchmark fixture lap-time label",
    "timestamp_ms": "synthetic: deterministic benchmark fixture timestamp",
}

BENCHMARK_LIMITATIONS = [
    "Committed benchmark fixture for deterministic development and promotion smoke gates.",
    "Not observed public telemetry and not suitable as production replay validation.",
]


def load_replay_events(path: str | Path) -> list[TelemetryEvent]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Replay dataset not found: {dataset_path}")
    if dataset_path.suffix.lower() == ".jsonl":
        return _load_jsonl(dataset_path)
    return _load_csv(dataset_path)


def replay_events(
    events: Iterable[TelemetryEvent],
    *,
    dataset_name: str,
    base_lap_time_s: float = 90.0,
    engine: InferenceEngine | None = None,
    fleet_intervals: list[dict] | None = None,
) -> ScenarioEvaluation:
    if engine is not None:
        active_engine = engine
    else:
        from f1_strategy.config import load_settings
        active_engine = InferenceEngine(
            settings=_replace(
                load_settings(),
                base_lap_time_s=base_lap_time_s,
                model_backend="hybrid",
                model_artifact_id="",
            )
        )
    event_list = list(events)
    _last_lap: dict[tuple[str, str], int] = {}
    errors: list[float] = []
    squared_errors: list[float] = []
    interval_widths: list[float] = []
    pit_target_errors: list[float] = []
    strategy_regrets: list[float] = []
    covered = 0
    lap_wear_by_stint: dict[tuple[str, str], dict[int, float]] = {}
    event_count = 0
    max_lap = max((event.lap for event in event_list), default=0)
    lap_times = {
        event.lap: event.lap_time_s for event in event_list if event.lap_time_s is not None
    }

    for event in event_list:
        event_count += 1
        lap_key = (event.session_id, event.car_id)
        if fleet_intervals is not None and _last_lap.get(lap_key) != event.lap:
            _last_lap[lap_key] = event.lap
            from f1_strategy.data_sources.openf1_intervals import build_fleet_state_for_lap
            fs = build_fleet_state_for_lap(
                fleet_intervals,
                event.session_id,
                event.lap,
                driver_number=event.car_id,
            )
            active_engine.update_fleet_state(fs)
        prediction = active_engine.ingest(event)
        stint_key = (event.session_id, event.car_id)
        lap_wear = lap_wear_by_stint.setdefault(stint_key, {})
        lap_wear[event.lap] = max(lap_wear.get(event.lap, 0.0), prediction.tire_wear_pct)

        if event.lap_time_s is None:
            continue
        actual_delta = event.lap_time_s - base_lap_time_s
        error = prediction.next_lap_delta_s - actual_delta
        errors.append(abs(error))
        squared_errors.append(error * error)
        interval_widths.append(prediction.uncertainty_high_s - prediction.uncertainty_low_s)
        if prediction.uncertainty_low_s <= actual_delta <= prediction.uncertainty_high_s:
            covered += 1
        remaining_laps = max(1, max_lap - event.lap)
        recommendation = active_engine.strategy(event.session_id, event.car_id, remaining_laps)
        oracle_pit_lap = _oracle_pit_lap(
            current_lap=event.lap,
            remaining_laps=remaining_laps,
            compound=event.compound,
            actual_delta_s=actual_delta,
            lap_times=lap_times,
            base_lap_time_s=base_lap_time_s,
        )
        pit_error = abs(recommendation.pit_window.target_lap - oracle_pit_lap)
        pit_target_errors.append(float(pit_error))
        strategy_regrets.append(
            _strategy_regret_s(pit_error, recommendation.energy_plan.expected_lap_gain_s)
        )

    observations = len(errors)
    monotonic_violations = 0
    for lap_wear in lap_wear_by_stint.values():
        ordered = [wear for _, wear in sorted(lap_wear.items())]
        monotonic_violations += sum(
            1 for index in range(1, len(ordered)) if ordered[index] + 1e-6 < ordered[index - 1]
        )
    return ScenarioEvaluation(
        scenario=dataset_name,
        laps=max((event.lap for event in event_list), default=0),
        compound="mixed",
        observations=observations,
        mae_lap_delta_s=mean(errors) if errors else 0.0,
        rmse_lap_delta_s=(mean(squared_errors) ** 0.5) if squared_errors else 0.0,
        mean_interval_width_s=mean(interval_widths) if interval_widths else 0.0,
        coverage_pct=(covered / observations * 100.0) if observations else 0.0,
        latency_p95_ms=active_engine.latency_p95_ms(),
        monotonic_wear_violations=monotonic_violations,
        calibration_error_pct=abs((covered / observations * 100.0) - 90.0)
        if observations
        else 0.0,
        pit_target_error_laps=mean(pit_target_errors) if pit_target_errors else 0.0,
        strategy_regret_s=mean(strategy_regrets) if strategy_regrets else 0.0,
        source="replay",
        event_count=event_count,
    )


def run_replay_evaluation(
    dataset_path: str | Path = DEFAULT_REPLAY_DATASET,
    *,
    gates: ReplayGateConfig | None = None,
    base_lap_time_s: float | None = None,
    engine: InferenceEngine | None = None,
    fleet_intervals_path: str | Path | None = None,
) -> ReplayEvaluationReport:
    path = Path(dataset_path)
    events = load_replay_events(path)
    if not events:
        raise ValueError(f"Replay dataset is empty: {path}")
    reference_lap_time_s = (
        base_lap_time_s
        if base_lap_time_s is not None
        else replay_reference_lap_time_s(path, default=90.0)
    )
    fleet_intervals: list[dict] | None = None
    from f1_strategy.data_sources.openf1_intervals import (
        fleet_intervals_path_for,
        load_fleet_intervals,
    )
    intervals_path = Path(fleet_intervals_path) if fleet_intervals_path else fleet_intervals_path_for(path)
    if intervals_path.exists():
        fleet_intervals = load_fleet_intervals(intervals_path)
    return evaluate_replay_event_list(
        events,
        dataset_name=path.stem,
        dataset_path=str(path),
        dataset_fingerprint=dataset_fingerprint(path),
        gates=gates,
        base_lap_time_s=reference_lap_time_s,
        engine=engine,
        fleet_intervals=fleet_intervals,
    )


def evaluate_replay_event_list(
    events: list[TelemetryEvent],
    *,
    dataset_name: str,
    dataset_path: str,
    dataset_fingerprint: str,
    gates: ReplayGateConfig | None = None,
    base_lap_time_s: float = 90.0,
    engine: InferenceEngine | None = None,
    fleet_intervals: list[dict] | None = None,
) -> ReplayEvaluationReport:
    gate_config = gates or ReplayGateConfig()
    provenance = replay_data_provenance(dataset_path)
    labeled = sum(1 for event in events if event.lap_time_s is not None)
    missing_target_pct = (len(events) - labeled) / len(events) * 100.0
    scenario = replay_events(
        events,
        dataset_name=dataset_name,
        base_lap_time_s=base_lap_time_s,
        engine=engine,
        fleet_intervals=fleet_intervals,
    )
    gate_results = {
        "mae_lap_delta": scenario.mae_lap_delta_s <= gate_config.max_mae_lap_delta_s,
        "coverage": scenario.coverage_pct >= gate_config.min_coverage_pct,
        "calibration": scenario.calibration_error_pct <= gate_config.max_calibration_error_pct,
        "sharpness": scenario.mean_interval_width_s <= gate_config.max_mean_interval_width_s,
        "latency": scenario.latency_p95_ms <= gate_config.max_latency_p95_ms,
        "monotonic_wear": (
            scenario.monotonic_wear_violations <= gate_config.max_monotonic_wear_violations
        ),
        "target_completeness": missing_target_pct <= gate_config.max_missing_target_pct,
        "sample_size": len(events) >= gate_config.min_event_count,
        "pit_decision": scenario.pit_target_error_laps <= gate_config.max_pit_target_error_laps,
        "strategy_regret": scenario.strategy_regret_s <= gate_config.max_strategy_regret_s,
    }
    if gate_config.require_observed_lap_time_labels:
        gate_results["observed_lap_time_labels"] = provenance.get("lap_time_label") == "observed"
    return ReplayEvaluationReport(
        version=APP_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_hash=feature_schema_hash(),
        dataset_path=dataset_path,
        dataset_fingerprint=dataset_fingerprint,
        session_count=len({event.session_id for event in events}),
        event_count=len(events),
        labeled_event_count=labeled,
        missing_target_pct=missing_target_pct,
        scenario=scenario,
        gates=gate_results,
        passed=all(gate_results.values()),
        data_provenance=provenance,
    )


def run_replay_suite(
    specs: list[ReplaySplitSpec] | None = None,
    *,
    gates: ReplayGateConfig | None = None,
    engine_factory: Callable[[float], InferenceEngine] | None = None,
    suite_name: str = "default",
) -> ReplaySuiteReport:
    selected = specs or DEFAULT_REPLAY_SUITE
    reports = []
    for spec in selected:
        events, fingerprint, dataset_path = _events_for_split(spec)
        split_engine = (
            engine_factory(spec.base_lap_time_s)
            if engine_factory is not None
            else _default_replay_engine(spec.base_lap_time_s)
        )
        reports.append(
            evaluate_replay_event_list(
                events,
                dataset_name=spec.name,
                dataset_path=dataset_path,
                dataset_fingerprint=fingerprint,
                gates=spec.gates or gates,
                base_lap_time_s=spec.base_lap_time_s,
                engine=split_engine,
            )
        )
    return ReplaySuiteReport(
        version=APP_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_hash=feature_schema_hash(),
        split_count=len(reports),
        passed=all(report.passed for report in reports),
        mean_mae_lap_delta_s=mean(
            report.scenario.mae_lap_delta_s for report in reports
        )
        if reports
        else 0.0,
        mean_coverage_pct=mean(report.scenario.coverage_pct for report in reports)
        if reports
        else 0.0,
        total_event_count=sum(report.event_count for report in reports),
        total_labeled_event_count=sum(report.labeled_event_count for report in reports),
        splits=reports,
        suite_name=suite_name,
    )


def run_benchmark_replay_suite(
    *,
    engine_factory: Callable[[float], InferenceEngine] | None = None,
) -> ReplaySuiteReport:
    return run_replay_suite(
        BENCHMARK_REPLAY_SUITE,
        engine_factory=engine_factory,
        suite_name="benchmark",
    )


def benchmark_manifest_payload(spec: ReplaySplitSpec) -> dict[str, Any]:
    if spec.dataset_path is None:
        raise ValueError(f"Benchmark split has no dataset path: {spec.name}")
    path = spec.dataset_path
    events = load_replay_events(path)
    suite_metadata = _benchmark_suite_metadata().get(str(path), {})
    return {
        "source": "benchmark-fixture",
        "generated_at": BENCHMARK_MANIFEST_GENERATED_AT,
        "description": suite_metadata.get("description", f"Deterministic benchmark fixture: {spec.name}."),
        "output": str(path),
        "dataset_fingerprint": dataset_fingerprint(path),
        "row_count": len(events),
        "lap_count": len({event.lap for event in events}),
        "session_count": len({event.session_id for event in events}),
        "reference_lap_time_s": spec.base_lap_time_s,
        "field_provenance": BENCHMARK_FIELD_PROVENANCE,
        "limitations": BENCHMARK_LIMITATIONS,
    }


def write_benchmark_manifests(*, check: bool = False) -> list[Path]:
    paths: list[Path] = []
    mismatches: list[str] = []
    for spec in BENCHMARK_REPLAY_SUITE:
        if spec.dataset_path is None:
            continue
        manifest_path = spec.dataset_path.with_suffix(spec.dataset_path.suffix + ".manifest.json")
        expected = benchmark_manifest_payload(spec)
        paths.append(manifest_path)
        if check:
            if not manifest_path.exists():
                mismatches.append(f"missing {manifest_path}")
                continue
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
            if actual != expected:
                mismatches.append(f"stale {manifest_path}")
            continue
        manifest_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    if mismatches:
        raise RuntimeError("Benchmark replay manifests are not current: " + ", ".join(mismatches))
    return paths


def _benchmark_suite_metadata() -> dict[str, dict[str, Any]]:
    path = BENCHMARK_REPLAY_DIR / "manifest.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {
        str(item.get("path")): dict(item)
        for item in payload.get("datasets", [])
        if item.get("path")
    }


def _default_replay_engine(base_lap_time_s: float) -> InferenceEngine:
    from f1_strategy.config import load_settings

    return InferenceEngine(
        settings=_replace(
            load_settings(),
            base_lap_time_s=base_lap_time_s,
            model_backend="hybrid",
            model_artifact_id="",
        )
    )


def replay_report_to_dict(report: ReplayEvaluationReport) -> dict:
    scenario = report.scenario
    return {
        "version": report.version,
        "feature_schema_version": report.feature_schema_version,
        "feature_schema_hash": report.feature_schema_hash,
        "dataset_path": report.dataset_path,
        "dataset_fingerprint": report.dataset_fingerprint,
        "session_count": report.session_count,
        "event_count": report.event_count,
        "labeled_event_count": report.labeled_event_count,
        "missing_target_pct": report.missing_target_pct,
        "passed": report.passed,
        "gates": report.gates,
        "data_provenance": report.data_provenance,
        "scenario": {
            "scenario": scenario.scenario,
            "source": scenario.source,
            "event_count": scenario.event_count,
            "laps": scenario.laps,
            "compound": scenario.compound,
            "observations": scenario.observations,
            "mae_lap_delta_s": scenario.mae_lap_delta_s,
            "rmse_lap_delta_s": scenario.rmse_lap_delta_s,
            "mean_interval_width_s": scenario.mean_interval_width_s,
            "coverage_pct": scenario.coverage_pct,
            "latency_p95_ms": scenario.latency_p95_ms,
            "monotonic_wear_violations": scenario.monotonic_wear_violations,
            "calibration_error_pct": scenario.calibration_error_pct,
            "pit_target_error_laps": scenario.pit_target_error_laps,
            "strategy_regret_s": scenario.strategy_regret_s,
        },
    }


def replay_suite_to_dict(report: ReplaySuiteReport) -> dict:
    return {
        "version": report.version,
        "feature_schema_version": report.feature_schema_version,
        "feature_schema_hash": report.feature_schema_hash,
        "suite_name": report.suite_name,
        "split_count": report.split_count,
        "passed": report.passed,
        "mean_mae_lap_delta_s": report.mean_mae_lap_delta_s,
        "mean_coverage_pct": report.mean_coverage_pct,
        "total_event_count": report.total_event_count,
        "total_labeled_event_count": report.total_labeled_event_count,
        "splits": [replay_report_to_dict(split) for split in report.splits],
    }


def replay_data_provenance(dataset_path: str | Path) -> dict[str, Any]:
    path_text = str(dataset_path)
    if path_text.startswith("generated:"):
        return {
            "source": "simulator",
            "manifest_available": False,
            "validation_signal": "synthetic",
            "lap_time_label": "synthetic",
            "observed_field_count": len(REPLAY_REQUIRED_COLUMNS) + len(REPLAY_OPTIONAL_COLUMNS),
            "derived_field_count": 0,
            "proxy_diagnostic_field_count": 0,
            "observed_fields": REPLAY_REQUIRED_COLUMNS + REPLAY_OPTIONAL_COLUMNS,
            "derived_fields": [],
            "proxy_diagnostic_fields": [],
            "promotion_signal_ready": False,
            "production_validation_ready": False,
            "reference_lap_time_s": None,
            "limitations": [
                "Generated simulator splits are deterministic development gates, not production validation."
            ],
        }

    path = Path(dataset_path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.exists():
        return {
            "source": "unknown",
            "manifest_available": False,
            "validation_signal": "unprovenanced",
            "lap_time_label": "unknown",
            "observed_field_count": 0,
            "derived_field_count": 0,
            "proxy_diagnostic_field_count": 0,
            "observed_fields": [],
            "derived_fields": [],
            "proxy_diagnostic_fields": [],
            "promotion_signal_ready": False,
            "production_validation_ready": False,
            "reference_lap_time_s": None,
            "limitations": [
                "Replay dataset has no sidecar manifest, so field provenance cannot be verified."
            ],
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    field_provenance = dict(manifest.get("field_provenance", {}))
    observed = _fields_with_prefix(field_provenance, "observed")
    derived = _fields_with_prefix(field_provenance, "derived")
    fallback = _fields_with_prefix(field_provenance, "fallback")
    proxy_fields = sorted(set(derived + fallback))
    lap_time_kind = _field_kind(field_provenance, "lap_time_s")
    observed_validation_fields = [
        name
        for name in [
            "speed_kph",
            "throttle",
            "brake",
            "track_temp_c",
            "air_temp_c",
            "humidity",
            "compound",
            "lap_time_s",
        ]
        if name in observed
    ]
    production_ready = lap_time_kind == "observed" and len(observed_validation_fields) >= 5
    return {
        "source": str(manifest.get("source", "manifested")),
        "manifest_available": True,
        "manifest_path": str(manifest_path),
        "validation_signal": "observed-public" if production_ready else "proxy-heavy",
        "lap_time_label": lap_time_kind or "unknown",
        "observed_field_count": len(observed),
        "derived_field_count": len(derived),
        "proxy_diagnostic_field_count": len(proxy_fields),
        "observed_fields": observed,
        "derived_fields": derived,
        "proxy_diagnostic_fields": proxy_fields,
        "observed_validation_fields": observed_validation_fields,
        "promotion_signal_ready": production_ready,
        "production_validation_ready": production_ready,
        "reference_lap_time_s": manifest.get("reference_lap_time_s"),
        "limitations": list(manifest.get("limitations", [])),
    }


def replay_reference_lap_time_s(path: str | Path, default: float = 90.0) -> float:
    dataset_path = Path(path)
    manifest_path = dataset_path.with_suffix(dataset_path.suffix + ".manifest.json")
    if not manifest_path.exists():
        return default
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = manifest.get("reference_lap_time_s")
        return float(value) if value is not None else default
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return default


def _fields_with_prefix(field_provenance: dict[str, Any], prefix: str) -> list[str]:
    return sorted(
        name
        for name, description in field_provenance.items()
        if _field_kind(field_provenance, name) == prefix
    )


def _field_kind(field_provenance: dict[str, Any], name: str) -> str:
    description = str(field_provenance.get(name, ""))
    return description.split(":", 1)[0].strip().lower()


def _events_for_split(spec: ReplaySplitSpec) -> tuple[list[TelemetryEvent], str, str]:
    if spec.dataset_path is not None:
        path = Path(spec.dataset_path)
        return load_replay_events(path), dataset_fingerprint(path), str(path)

    simulator = RaceSimulator(
        SimulationConfig(
            session_id=f"replay-{spec.name}",
            car_id="car-replay",
            laps=spec.laps,
            seed=spec.seed,
            compound=spec.compound,
            base_lap_time_s=spec.base_lap_time_s,
        )
    )
    events = []
    raw_events = simulator.events()
    lap_times = {event.lap: event.lap_time_s for event in raw_events if event.lap_time_s is not None}
    for event in raw_events:
        payload = to_jsonable(event)
        payload["lap_time_s"] = lap_times.get(event.lap, payload.get("lap_time_s"))
        if spec.track_temp_offset_c:
            payload["track_temp_c"] = float(payload["track_temp_c"]) + spec.track_temp_offset_c
            payload["tire_temp_fl"] = float(payload["tire_temp_fl"]) + spec.track_temp_offset_c * 0.30
            payload["tire_temp_fr"] = float(payload["tire_temp_fr"]) + spec.track_temp_offset_c * 0.30
            payload["tire_temp_rl"] = float(payload["tire_temp_rl"]) + spec.track_temp_offset_c * 0.22
            payload["tire_temp_rr"] = float(payload["tire_temp_rr"]) + spec.track_temp_offset_c * 0.22
        if spec.dirty_air:
            payload["speed_kph"] = float(payload["speed_kph"]) - 8.0
            payload["tire_temp_fl"] = float(payload["tire_temp_fl"]) + 3.0
            payload["tire_temp_fr"] = float(payload["tire_temp_fr"]) + 3.0
            payload["brake_temp"] = float(payload["brake_temp"]) + 45.0
            payload["slip_angle"] = float(payload["slip_angle"]) + 0.8
        events.append(telemetry_from_dict(payload))
    fingerprint = _event_fingerprint(events, spec.name)
    return events, fingerprint, f"generated:{spec.name}"


def _event_fingerprint(events: list[TelemetryEvent], name: str) -> str:
    import hashlib

    payload = {
        "name": name,
        "feature_schema_hash": feature_schema_hash(),
        "events": [to_jsonable(event) for event in events],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def dataset_fingerprint(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_replay_payload(payload: dict) -> dict:
    normalized = dict(payload)
    for ignored in REPLAY_IGNORED_COLUMNS:
        normalized.pop(ignored, None)
    missing = [name for name in REPLAY_REQUIRED_COLUMNS if name not in normalized]
    if missing:
        raise ValueError(f"Replay row is missing required columns: {', '.join(missing)}")
    for key, value in list(normalized.items()):
        if value == "":
            normalized[key] = None
    for name in ["lap", "sector", "timestamp_ms"]:
        if normalized.get(name) is not None:
            normalized[name] = int(normalized[name])
    for name in [
        "speed_kph",
        "throttle",
        "brake",
        "steering_angle",
        "tire_temp_fl",
        "tire_temp_fr",
        "tire_temp_rl",
        "tire_temp_rr",
        "brake_temp",
        "slip_angle",
        "lateral_g",
        "ers_soc",
        "ers_deployment_kw",
        "fuel_kg",
        "track_temp_c",
        "air_temp_c",
        "humidity",
        "lap_time_s",
    ]:
        if normalized.get(name) is not None:
            normalized[name] = float(normalized[name])
    return normalized


def _load_csv(path: Path) -> list[TelemetryEvent]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Replay CSV has no header: {path}")
        unknown = [
            name
            for name in reader.fieldnames
            if name not in REPLAY_REQUIRED_COLUMNS
            and name not in REPLAY_OPTIONAL_COLUMNS
            and name not in REPLAY_IGNORED_COLUMNS
        ]
        if unknown:
            raise ValueError(f"Replay CSV has unsupported columns: {', '.join(unknown)}")
        return [telemetry_from_dict(validate_replay_payload(row)) for row in reader]


def _load_jsonl(path: Path) -> list[TelemetryEvent]:
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc
            events.append(telemetry_from_dict(validate_replay_payload(payload)))
    return events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a model by replaying recorded telemetry.")
    parser.add_argument("--dataset", default=str(DEFAULT_REPLAY_DATASET))
    parser.add_argument("--suite", action="store_true", help="Run the default named replay suite.")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the committed benchmark replay suite with per-slice gates.",
    )
    parser.add_argument(
        "--write-benchmark-manifests",
        action="store_true",
        help="Regenerate sidecar manifests for committed benchmark replay fixtures.",
    )
    parser.add_argument(
        "--check-benchmark-manifests",
        action="store_true",
        help="Fail if committed benchmark replay sidecar manifests are stale.",
    )
    parser.add_argument("--format", choices=["json"], default="json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.write_benchmark_manifests:
        paths = write_benchmark_manifests()
        print(json.dumps({"written": [str(path) for path in paths]}, indent=2, sort_keys=True))
        return
    if args.check_benchmark_manifests:
        paths = write_benchmark_manifests(check=True)
        print(json.dumps({"checked": [str(path) for path in paths]}, indent=2, sort_keys=True))
        return
    if args.benchmark:
        print(
            json.dumps(
                replay_suite_to_dict(run_benchmark_replay_suite()),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.suite:
        print(json.dumps(replay_suite_to_dict(run_replay_suite()), indent=2, sort_keys=True))
    else:
        report = run_replay_evaluation(args.dataset)
        print(json.dumps(replay_report_to_dict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
