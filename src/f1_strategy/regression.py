from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from f1_strategy.config import Settings, load_settings
from f1_strategy.engine import InferenceEngine
from f1_strategy.persistence import NullPersistence
from f1_strategy.simulation import RaceSimulator, SimulationConfig


@dataclass(frozen=True)
class RegressionResult:
    name: str
    passed: bool
    value: float
    threshold: float


class RegressionSuite:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()

    def run(self) -> list[RegressionResult]:
        engine = InferenceEngine(persistence=NullPersistence())
        simulator = RaceSimulator(SimulationConfig(laps=18, seed=11))
        predictions = [engine.ingest(event) for event in simulator.events()]
        lap_predictions = [p for index, p in enumerate(predictions, start=1) if index % 3 == 0]
        deltas = [p.next_lap_delta_s for p in lap_predictions]
        widths = [p.uncertainty_high_s - p.uncertainty_low_s for p in lap_predictions]

        stability = max(
            abs(deltas[index] - deltas[index - 1]) for index in range(1, len(deltas))
        )
        calibration_width = mean(widths)
        latency_p95 = engine.latency_p95_ms()
        monotonic_wear_violations = sum(
            1
            for index in range(1, len(lap_predictions))
            if lap_predictions[index].tire_wear_pct + 1e-6
            < lap_predictions[index - 1].tire_wear_pct
        )

        return [
            RegressionResult(
                "latency_p95_ms",
                latency_p95 <= self.settings.target_latency_ms,
                latency_p95,
                self.settings.target_latency_ms,
            ),
            RegressionResult(
                "temporal_stability_s",
                stability <= self.settings.max_temporal_oscillation_s,
                stability,
                self.settings.max_temporal_oscillation_s,
            ),
            RegressionResult(
                "calibration_interval_width_s",
                0.20 <= calibration_width <= self.settings.max_calibration_width_s,
                calibration_width,
                self.settings.max_calibration_width_s,
            ),
            RegressionResult(
                "monotonic_wear_violations",
                monotonic_wear_violations == 0,
                float(monotonic_wear_violations),
                0.0,
            ),
        ]


def main() -> None:
    results = RegressionSuite().run()
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.name}: value={result.value:.4f} threshold={result.threshold:.4f}")
    if not all(result.passed for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
