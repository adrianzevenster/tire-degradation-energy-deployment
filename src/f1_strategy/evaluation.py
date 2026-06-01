from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from f1_strategy.domain import TireCompound
from f1_strategy.engine import InferenceEngine
from f1_strategy.metadata import APP_VERSION
from f1_strategy.models import (
    FEATURE_SCHEMA_VERSION,
    ModelConfig,
    create_serving_model,
    feature_schema_hash,
)
from f1_strategy.simulation import RaceSimulator, SimulationConfig


@dataclass(frozen=True)
class EvaluationScenario:
    name: str
    laps: int
    seed: int
    compound: TireCompound
    base_lap_time_s: float = 90.0


@dataclass(frozen=True)
class ScenarioEvaluation:
    scenario: str
    laps: int
    compound: str
    observations: int
    mae_lap_delta_s: float
    rmse_lap_delta_s: float
    mean_interval_width_s: float
    coverage_pct: float
    latency_p95_ms: float
    monotonic_wear_violations: int


@dataclass(frozen=True)
class EvaluationReport:
    version: str
    feature_schema_version: str
    feature_schema_hash: str
    scenarios: list[ScenarioEvaluation]

    @property
    def mean_mae_lap_delta_s(self) -> float:
        return mean(item.mae_lap_delta_s for item in self.scenarios) if self.scenarios else 0.0

    @property
    def mean_coverage_pct(self) -> float:
        return mean(item.coverage_pct for item in self.scenarios) if self.scenarios else 0.0


DEFAULT_SCENARIOS = [
    EvaluationScenario("medium-baseline", laps=18, seed=21, compound=TireCompound.MEDIUM),
    EvaluationScenario("soft-degradation", laps=18, seed=22, compound=TireCompound.SOFT),
    EvaluationScenario("hard-long-run", laps=24, seed=23, compound=TireCompound.HARD),
    EvaluationScenario("intermediate-pace", laps=16, seed=24, compound=TireCompound.INTERMEDIATE),
]


def evaluate_scenario(
    scenario: EvaluationScenario,
    model_backend: str | None = None,
    model_paths: dict[str, str] | None = None,
) -> ScenarioEvaluation:
    engine = _evaluation_engine(model_backend=model_backend, model_paths=model_paths)
    simulator = RaceSimulator(
        SimulationConfig(
            session_id=f"eval-{scenario.name}",
            car_id="car-eval",
            laps=scenario.laps,
            seed=scenario.seed,
            compound=scenario.compound,
            base_lap_time_s=scenario.base_lap_time_s,
        )
    )
    errors: list[float] = []
    squared_errors: list[float] = []
    interval_widths: list[float] = []
    covered = 0
    lap_wear: list[float] = []

    for event in simulator.events():
        prediction = engine.ingest(event)
        if event.lap_time_s is None:
            continue
        actual_delta = event.lap_time_s - scenario.base_lap_time_s
        error = prediction.next_lap_delta_s - actual_delta
        errors.append(abs(error))
        squared_errors.append(error * error)
        interval_widths.append(prediction.uncertainty_high_s - prediction.uncertainty_low_s)
        if prediction.uncertainty_low_s <= actual_delta <= prediction.uncertainty_high_s:
            covered += 1
        lap_wear.append(prediction.tire_wear_pct)

    monotonic_violations = sum(
        1 for index in range(1, len(lap_wear)) if lap_wear[index] + 1e-6 < lap_wear[index - 1]
    )
    observations = len(errors)

    return ScenarioEvaluation(
        scenario=scenario.name,
        laps=scenario.laps,
        compound=scenario.compound.value,
        observations=observations,
        mae_lap_delta_s=mean(errors) if errors else 0.0,
        rmse_lap_delta_s=(mean(squared_errors) ** 0.5) if squared_errors else 0.0,
        mean_interval_width_s=mean(interval_widths) if interval_widths else 0.0,
        coverage_pct=(covered / observations * 100.0) if observations else 0.0,
        latency_p95_ms=engine.latency_p95_ms(),
        monotonic_wear_violations=monotonic_violations,
    )


def run_evaluation(
    scenarios: list[EvaluationScenario] | None = None,
    model_backend: str | None = None,
    model_paths: dict[str, str] | None = None,
) -> EvaluationReport:
    selected = scenarios or DEFAULT_SCENARIOS
    return EvaluationReport(
        version=APP_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_hash=feature_schema_hash(),
        scenarios=[
            evaluate_scenario(
                scenario,
                model_backend=model_backend,
                model_paths=model_paths,
            )
            for scenario in selected
        ],
    )


def _evaluation_engine(
    model_backend: str | None = None,
    model_paths: dict[str, str] | None = None,
) -> InferenceEngine:
    if model_backend is None:
        return InferenceEngine()
    paths = model_paths or {}
    config = ModelConfig()
    model = create_serving_model(
        config=config,
        backend=model_backend,
        xgboost_model_path=paths.get("xgboost", "models/xgboost_lap_delta.json"),
        lightgbm_model_path=paths.get("lightgbm", "models/lightgbm_lap_delta.txt"),
        catboost_model_path=paths.get("catboost", "models/catboost_lap_delta.cbm"),
        sequence_model_path=paths.get("sequence", "models/sequence_lap_delta.pt"),
    )
    return InferenceEngine(model=model)


def render_markdown(report: EvaluationReport) -> str:
    lines = [
        "# Model Evaluation Report",
        "",
        f"- Version: `{report.version}`",
        f"- Feature schema: `{report.feature_schema_version}`",
        f"- Feature schema hash: `{report.feature_schema_hash}`",
        f"- Scenarios: `{len(report.scenarios)}`",
        f"- Mean MAE lap delta: `{report.mean_mae_lap_delta_s:.4f}s`",
        f"- Mean interval coverage: `{report.mean_coverage_pct:.1f}%`",
        "",
        "| Scenario | Compound | Laps | MAE (s) | RMSE (s) | Coverage | "
        "Width (s) | p95 latency (ms) | Wear violations |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report.scenarios:
        lines.append(
            "| "
            f"{item.scenario} | {item.compound} | {item.laps} | "
            f"{item.mae_lap_delta_s:.4f} | {item.rmse_lap_delta_s:.4f} | "
            f"{item.coverage_pct:.1f}% | {item.mean_interval_width_s:.4f} | "
            f"{item.latency_p95_ms:.4f} | {item.monotonic_wear_violations} |"
        )
    lines.append("")
    return "\n".join(lines)


def report_to_dict(report: EvaluationReport) -> dict:
    return {
        "version": report.version,
        "feature_schema_version": report.feature_schema_version,
        "feature_schema_hash": report.feature_schema_hash,
        "scenario_count": len(report.scenarios),
        "mean_mae_lap_delta_s": report.mean_mae_lap_delta_s,
        "mean_coverage_pct": report.mean_coverage_pct,
        "scenarios": [
            {
                "scenario": item.scenario,
                "laps": item.laps,
                "compound": item.compound,
                "observations": item.observations,
                "mae_lap_delta_s": item.mae_lap_delta_s,
                "rmse_lap_delta_s": item.rmse_lap_delta_s,
                "mean_interval_width_s": item.mean_interval_width_s,
                "coverage_pct": item.coverage_pct,
                "latency_p95_ms": item.latency_p95_ms,
                "monotonic_wear_violations": item.monotonic_wear_violations,
            }
            for item in report.scenarios
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic model evaluation scenarios.")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--output", default=None, help="Optional report output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_evaluation()
    content = (
        json.dumps(report_to_dict(report), indent=2, sort_keys=True)
        if args.format == "json"
        else render_markdown(report)
    )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    else:
        print(content)


if __name__ == "__main__":
    main()
