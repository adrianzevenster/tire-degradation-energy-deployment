from __future__ import annotations

from dataclasses import dataclass
from time import time

from f1_strategy.domain import TelemetryEvent
from f1_strategy.simulation import RaceSimulator, SimulationConfig


@dataclass
class LiveSimulationState:
    events: list[TelemetryEvent]
    index: int = 0
    running: bool = False
    session_id: str = ""

    @property
    def complete(self) -> bool:
        return self.index >= len(self.events)


class LiveSimulationManager:
    def __init__(self) -> None:
        self._state = LiveSimulationState(events=[])

    def start(
        self,
        laps: int = 18,
        seed: int = 7,
        session_id: str | None = None,
    ) -> LiveSimulationState:
        run_id = session_id or f"sim-race-{int(time() * 1000)}"
        simulator = RaceSimulator(SimulationConfig(session_id=run_id, laps=laps, seed=seed))
        self._state = LiveSimulationState(
            events=simulator.events(),
            running=True,
            session_id=run_id,
        )
        return self._state

    def stop(self) -> LiveSimulationState:
        self._state.running = False
        return self._state

    def reset(self) -> LiveSimulationState:
        self._state = LiveSimulationState(events=[])
        return self._state

    def tick(self, batch_size: int = 1) -> list[TelemetryEvent]:
        if not self._state.running:
            return []
        start = self._state.index
        end = min(len(self._state.events), start + max(1, batch_size))
        events = self._state.events[start:end]
        self._state.index = end
        if self._state.complete:
            self._state.running = False
        return events

    def status(self) -> dict[str, int | bool | str]:
        total = len(self._state.events)
        return {
            "running": self._state.running,
            "complete": self._state.complete if total else False,
            "session_id": self._state.session_id,
            "index": self._state.index,
            "total": total,
            "remaining": max(0, total - self._state.index),
        }
