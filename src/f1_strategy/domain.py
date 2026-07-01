from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any


class TireCompound(str, Enum):
    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"
    INTERMEDIATE = "intermediate"
    WET = "wet"


@dataclass(frozen=True)
class TelemetryEvent:
    session_id: str
    car_id: str
    lap: int
    sector: int
    speed_kph: float
    throttle: float
    brake: float
    steering_angle: float
    tire_temp_fl: float
    tire_temp_fr: float
    tire_temp_rl: float
    tire_temp_rr: float
    brake_temp: float
    slip_angle: float
    lateral_g: float
    ers_soc: float
    ers_deployment_kw: float
    fuel_kg: float
    track_temp_c: float
    air_temp_c: float
    humidity: float
    compound: TireCompound
    lap_time_s: float | None = None
    timestamp_ms: int = field(default_factory=lambda: int(time() * 1000))
    circuit: str = ""


@dataclass(frozen=True)
class OnlineFeatures:
    session_id: str
    car_id: str
    lap: int
    compound: TireCompound
    circuit: str
    tire_age_laps: int
    mean_tire_temp: float
    tire_temp_gradient: float
    brake_heat_index: float
    driver_aggression: float
    steering_entropy: float
    cumulative_tire_load: float
    degradation_acceleration: float
    ers_efficiency: float
    ers_soc: float
    track_temp_c: float
    fuel_kg: float
    rolling_lap_time_s: float
    dirty_air_risk: float
    circuit_base_lap_s: float = 90.0
    stint_lap_delta_s: float = 0.0
    lap_to_lap_delta_s: float = 0.0
    # Fleet context — not in the ML feature vector, used by the strategy optimizer only.
    fleet_gap_ahead_s: float = 999.0
    fleet_gap_behind_s: float = 999.0
    fleet_position: int = 10
    fleet_competitor_tire_age: int = 0
    fleet_competitor_compound: TireCompound = TireCompound.MEDIUM


@dataclass(frozen=True)
class Prediction:
    session_id: str
    car_id: str
    lap: int
    tire_wear_pct: float
    remaining_tire_life_laps: float
    grip_loss_pct: float
    overheating_probability: float
    cliff_probability: float
    brake_temp_next_lap_c: float
    ers_efficiency: float
    next_lap_delta_s: float
    uncertainty_low_s: float
    uncertainty_high_s: float
    model_backend: str = "unknown"
    model_artifact_id: str = "unregistered"
    model_feature_schema_hash: str = "unknown"
    app_version: str = "unknown"
    build_sha: str = "unknown"


@dataclass(frozen=True)
class EnergyPlan:
    sector_deployment_kw: dict[int, float]
    expected_lap_gain_s: float
    ending_soc: float


@dataclass(frozen=True)
class PitWindow:
    earliest_lap: int
    target_lap: int
    latest_lap: int
    undercut_success_probability: float
    safety_car_sensitivity: float
    undercut_window_laps: int = 0
    overcut_window_laps: int = 0
    competitor_tire_delta_laps: int = 0


@dataclass(frozen=True)
class StrategyRecommendation:
    session_id: str
    car_id: str
    prediction: Prediction
    pit_window: PitWindow
    energy_plan: EnergyPlan
    pace_target_delta_s: float
    reasons: list[str]
    cliff_lap_estimate: int = 0


@dataclass(frozen=True)
class DriftReport:
    drifted: bool
    feature_scores: dict[str, float]
    alerts: list[str]


class FleetState:
    """Mutable per-session state tracking inter-car gaps and competitor tire state.

    Updated each lap by whatever data source has race-control timing (OpenF1 live,
    timing CSV, or simulation). Consumed by StrategyOptimizer to compute undercut
    probability and dirty-air risk against real gap data instead of telemetry proxies.
    """

    _UNKNOWN_GAP = 999.0

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._gaps_ahead: dict[str, float] = {}
        self._gaps_behind: dict[str, float] = {}
        self._positions: dict[str, int] = {}
        self._competitor_tire_age: dict[str, int] = {}
        self._competitor_compound: dict[str, TireCompound] = {}

    def update(
        self,
        car_id: str,
        *,
        position: int = 10,
        gap_ahead_s: float = _UNKNOWN_GAP,
        gap_behind_s: float = _UNKNOWN_GAP,
        competitor_tire_age: int = 0,
        competitor_compound: TireCompound = TireCompound.MEDIUM,
    ) -> None:
        self._positions[car_id] = position
        self._gaps_ahead[car_id] = gap_ahead_s
        self._gaps_behind[car_id] = gap_behind_s
        self._competitor_tire_age[car_id] = competitor_tire_age
        self._competitor_compound[car_id] = competitor_compound

    def gap_ahead_s(self, car_id: str) -> float:
        return self._gaps_ahead.get(car_id, self._UNKNOWN_GAP)

    def gap_behind_s(self, car_id: str) -> float:
        return self._gaps_behind.get(car_id, self._UNKNOWN_GAP)

    def position(self, car_id: str) -> int:
        return self._positions.get(car_id, 10)

    def competitor_tire_age(self, car_id: str) -> int:
        return self._competitor_tire_age.get(car_id, 0)

    def competitor_compound(self, car_id: str) -> TireCompound:
        return self._competitor_compound.get(car_id, TireCompound.MEDIUM)

    def undercut_threat(self, car_id: str) -> float:
        """Probability that the car behind can gain time via an undercut pit stop."""
        gap = self._gaps_behind.get(car_id, self._UNKNOWN_GAP)
        own_age = self._competitor_tire_age.get(car_id, 0)
        if gap >= self._UNKNOWN_GAP:
            return 0.0
        # Threat peaks when gap < pit stop loss and car behind has older tires.
        pit_loss_s = 22.0
        gap_factor = max(0.0, 1.0 - gap / pit_loss_s)
        age_factor = min(1.0, own_age / 25.0)
        return min(1.0, gap_factor * 0.65 + age_factor * 0.35)

    def as_features(self, car_id: str) -> dict[str, float | int | TireCompound]:
        return {
            "fleet_gap_ahead_s": self.gap_ahead_s(car_id),
            "fleet_gap_behind_s": self.gap_behind_s(car_id),
            "fleet_position": self.position(car_id),
            "fleet_competitor_tire_age": self.competitor_tire_age(car_id),
            "fleet_competitor_compound": self.competitor_compound(car_id),
        }


JsonDict = dict[str, Any]
