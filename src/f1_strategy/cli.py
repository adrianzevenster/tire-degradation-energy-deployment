from __future__ import annotations

import argparse
import json

from f1_strategy.engine import InferenceEngine
from f1_strategy.metadata import APP_VERSION
from f1_strategy.serialization import to_jsonable
from f1_strategy.simulation import RaceSimulator, SimulationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic F1 telemetry simulation.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {APP_VERSION}")
    parser.add_argument("--laps", type=int, default=10)
    parser.add_argument("--remaining-laps", type=int, default=35)
    return parser


def simulate_main() -> None:
    args = build_parser().parse_args()
    engine = InferenceEngine()
    simulator = RaceSimulator(SimulationConfig(laps=args.laps))
    prediction = None
    for event in simulator.events():
        prediction = engine.ingest(event)
    if prediction is None:
        raise RuntimeError("Simulation produced no telemetry")
    strategy = engine.strategy(prediction.session_id, prediction.car_id, args.remaining_laps)
    print(json.dumps(to_jsonable(strategy), indent=2, sort_keys=True))


if __name__ == "__main__":
    simulate_main()
