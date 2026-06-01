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


@dataclass(frozen=True)
class OnlineFeatures:
    session_id: str
    car_id: str
    lap: int
    compound: TireCompound
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


@dataclass(frozen=True)
class StrategyRecommendation:
    session_id: str
    car_id: str
    prediction: Prediction
    pit_window: PitWindow
    energy_plan: EnergyPlan
    pace_target_delta_s: float
    reasons: list[str]


@dataclass(frozen=True)
class DriftReport:
    drifted: bool
    feature_scores: dict[str, float]
    alerts: list[str]


JsonDict = dict[str, Any]
