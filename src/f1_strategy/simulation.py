from __future__ import annotations

from dataclasses import dataclass
from random import Random

from f1_strategy.domain import TelemetryEvent, TireCompound


@dataclass(frozen=True)
class SimulationConfig:
    session_id: str = "sim-race"
    car_id: str = "car-16"
    laps: int = 20
    sectors_per_lap: int = 3
    seed: int = 7
    compound: TireCompound = TireCompound.MEDIUM
    base_lap_time_s: float = 90.0
    circuit: str = "synthetic"


class RaceSimulator:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self._rng = Random(self.config.seed)

    def events(self) -> list[TelemetryEvent]:
        rows: list[TelemetryEvent] = []
        fuel = 70.0
        ers_soc = 0.82
        tire_heat = 88.0
        for lap in range(1, self.config.laps + 1):
            aggression = 0.65 + 0.18 * self._rng.random()
            track_temp = 37.0 + lap * 0.08 + self._rng.uniform(-1.2, 1.2)
            lap_degradation = lap * 0.045 + max(track_temp - 40.0, 0.0) * 0.02
            lap_time = (
                self.config.base_lap_time_s
                + lap_degradation
                + aggression * 0.18
                - max(0.0, 70.0 - fuel) * 0.020
                + self._rng.uniform(-0.12, 0.12)
            )
            for sector in range(1, self.config.sectors_per_lap + 1):
                braking = min(1.0, 0.45 + 0.12 * sector + self._rng.uniform(-0.08, 0.08))
                steering = self._rng.uniform(-24.0, 24.0) * (1.0 + aggression * 0.15)
                slip = self._rng.uniform(1.0, 5.0) + lap * 0.035 + aggression * 0.8
                tire_heat += braking * 0.22 + slip * 0.035 + max(track_temp - 36, 0) * 0.015
                ers_deploy = max(0.0, 70.0 + self._rng.uniform(-12.0, 12.0) - sector * 4.0)
                ers_soc = min(1.0, max(0.05, ers_soc - ers_deploy / 1500.0 + 0.025))
                rows.append(
                    TelemetryEvent(
                        session_id=self.config.session_id,
                        car_id=self.config.car_id,
                        circuit=self.config.circuit,
                        lap=lap,
                        sector=sector,
                        speed_kph=235.0 + self._rng.uniform(-18.0, 22.0),
                        throttle=min(1.0, 0.68 + self._rng.random() * 0.25),
                        brake=braking,
                        steering_angle=steering,
                        tire_temp_fl=tire_heat + self._rng.uniform(-2.5, 2.5),
                        tire_temp_fr=tire_heat + self._rng.uniform(-2.5, 2.5),
                        tire_temp_rl=tire_heat - 3.0 + self._rng.uniform(-2.5, 2.5),
                        tire_temp_rr=tire_heat - 2.5 + self._rng.uniform(-2.5, 2.5),
                        brake_temp=520.0 + braking * 390.0 + self._rng.uniform(-30.0, 30.0),
                        slip_angle=slip,
                        lateral_g=2.1 + aggression * 1.1 + self._rng.uniform(-0.2, 0.2),
                        ers_soc=ers_soc,
                        ers_deployment_kw=ers_deploy,
                        fuel_kg=fuel,
                        track_temp_c=track_temp,
                        air_temp_c=27.0 + self._rng.uniform(-0.8, 0.8),
                        humidity=0.45 + self._rng.uniform(-0.06, 0.06),
                        compound=self.config.compound,
                        lap_time_s=lap_time if sector == self.config.sectors_per_lap else None,
                        timestamp_ms=(lap * 10_000) + sector * 3_000,
                    )
                )
            fuel = max(0.0, fuel - 1.8)
        return rows
