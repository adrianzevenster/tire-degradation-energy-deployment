from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable

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

REPLAY_OPTIONAL_COLUMNS = ["lap_time_s", "timestamp_ms"]
DEFAULT_REPLAY_DATASET = Path("examples/replay_telemetry.csv")


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
) -> ScenarioEvaluation:
    active_engine = engine or InferenceEngine()
    event_list = list(events)
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
    base_lap_time_s: float = 90.0,
    engine: InferenceEngine | None = None,
) -> ReplayEvaluationReport:
    path = Path(dataset_path)
    events = load_replay_events(path)
    if not events:
        raise ValueError(f"Replay dataset is empty: {path}")
    return evaluate_replay_event_list(
        events,
        dataset_name=path.stem,
        dataset_path=str(path),
        dataset_fingerprint=dataset_fingerprint(path),
        gates=gates,
        base_lap_time_s=base_lap_time_s,
        engine=engine,
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
) -> ReplayEvaluationReport:
    gate_config = gates or ReplayGateConfig()
    labeled = sum(1 for event in events if event.lap_time_s is not None)
    missing_target_pct = (len(events) - labeled) / len(events) * 100.0
    scenario = replay_events(
        events,
        dataset_name=dataset_name,
        base_lap_time_s=base_lap_time_s,
        engine=engine,
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
    )


def run_replay_suite(
    specs: list[ReplaySplitSpec] | None = None,
    *,
    gates: ReplayGateConfig | None = None,
    engine_factory: Callable[[], InferenceEngine] | None = None,
) -> ReplaySuiteReport:
    selected = specs or DEFAULT_REPLAY_SUITE
    reports = []
    for spec in selected:
        events, fingerprint, dataset_path = _events_for_split(spec)
        reports.append(
            evaluate_replay_event_list(
                events,
                dataset_name=spec.name,
                dataset_path=dataset_path,
                dataset_fingerprint=fingerprint,
                gates=gates,
                base_lap_time_s=spec.base_lap_time_s,
                engine=engine_factory() if engine_factory is not None else None,
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
        "split_count": report.split_count,
        "passed": report.passed,
        "mean_mae_lap_delta_s": report.mean_mae_lap_delta_s,
        "mean_coverage_pct": report.mean_coverage_pct,
        "total_event_count": report.total_event_count,
        "total_labeled_event_count": report.total_labeled_event_count,
        "splits": [replay_report_to_dict(split) for split in report.splits],
    }


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
    missing = [name for name in REPLAY_REQUIRED_COLUMNS if name not in payload]
    if missing:
        raise ValueError(f"Replay row is missing required columns: {', '.join(missing)}")
    normalized = dict(payload)
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
            if name not in REPLAY_REQUIRED_COLUMNS and name not in REPLAY_OPTIONAL_COLUMNS
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
    parser.add_argument("--format", choices=["json"], default="json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.suite:
        print(json.dumps(replay_suite_to_dict(run_replay_suite()), indent=2, sort_keys=True))
    else:
        report = run_replay_evaluation(args.dataset)
        print(json.dumps(replay_report_to_dict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
