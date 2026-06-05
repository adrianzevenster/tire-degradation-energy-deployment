from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from math import exp, sqrt
from pathlib import Path
from statistics import mean
from typing import Protocol

from f1_strategy.domain import OnlineFeatures, Prediction, TireCompound


COMPOUND_LIFE = {
    TireCompound.SOFT: 18.0,
    TireCompound.MEDIUM: 28.0,
    TireCompound.HARD: 40.0,
    TireCompound.INTERMEDIATE: 24.0,
    TireCompound.WET: 30.0,
}

COMPOUND_PACE = {
    TireCompound.SOFT: -0.55,
    TireCompound.MEDIUM: 0.0,
    TireCompound.HARD: 0.45,
    TireCompound.INTERMEDIATE: 1.8,
    TireCompound.WET: 3.0,
}

def _circuit_enc(typical_lap_s: float, base: float = 90.0, scale: float = 15.0) -> float:
    return round((typical_lap_s - base) / scale, 3)


# Scaled lap delta from 90s base: (typical_race_lap_s - 90) / 15
# Creates a monotonic encoding so XGBoost needs fewer splits to learn circuit offsets.
CIRCUIT_ENCODING: dict[str, float] = {
    "synthetic":   _circuit_enc(90),    #  0.000
    "monaco":      _circuit_enc(78.4),  # -0.773
    "imola":       _circuit_enc(82.0),  # -0.533
    "monza":       _circuit_enc(84.3),  # -0.380
    "australia":   _circuit_enc(84.5),  # -0.367
    "spa":         _circuit_enc(88.5),  # -0.100
    "zandvoort":   _circuit_enc(89.5),  # -0.033
    "canada":      _circuit_enc(90.5),  #  0.033
    "saudi-arabia":_circuit_enc(91.5),  #  0.100
    "miami":       _circuit_enc(92.0),  #  0.133
    "usa":         _circuit_enc(93.0),  #  0.200
    "japan":       _circuit_enc(93.5),  #  0.233
    "spain":       _circuit_enc(94.0),  #  0.267
    "silverstone": _circuit_enc(94.2),  #  0.280
    "austria":     _circuit_enc(94.5),  #  0.300
    "baku":        _circuit_enc(95.0),  #  0.333
    "bahrain":     _circuit_enc(95.8),  #  0.387
    "china":       _circuit_enc(96.0),  #  0.400
    "abu-dhabi":   _circuit_enc(96.0),  #  0.400
    "brazil":      _circuit_enc(97.0),  #  0.467
    "singapore":   _circuit_enc(97.4),  #  0.493
    "las-vegas":   _circuit_enc(97.5),  #  0.500
    "mexico":      _circuit_enc(98.5),  #  0.567
    "hungary":     _circuit_enc(99.0),  #  0.600
    "qatar":       _circuit_enc(100.0), #  0.667
}

FEATURE_NAMES = [
    "bias",
    "circuit",
    "rolling_lap_time_s",
    "tire_age_laps",
    "mean_tire_temp",
    "tire_temp_gradient",
    "brake_heat_index",
    "driver_aggression",
    "steering_entropy",
    "cumulative_tire_load",
    "degradation_acceleration",
    "ers_efficiency",
    "ers_soc",
    "track_temp_c",
    "fuel_kg",
    "dirty_air_risk",
    "circuit_base_lap_s",
    *[f"compound_{compound.value}" for compound in TireCompound],
]

FEATURE_SCHEMA_VERSION = "online-features-v3"


def feature_schema_hash(feature_names: list[str] | None = None) -> str:
    names = feature_names or FEATURE_NAMES
    payload = json.dumps(
        {"version": FEATURE_SCHEMA_VERSION, "features": names},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_manifest_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    return path.with_suffix(path.suffix + ".manifest.json")


def model_manifest(
    backend: str,
    training_rows: int | None = None,
    feature_names: list[str] | None = None,
) -> dict[str, object]:
    names = feature_names or FEATURE_NAMES
    manifest: dict[str, object] = {
        "backend": backend,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": feature_schema_hash(names),
        "feature_names": names,
    }
    if training_rows is not None:
        manifest["training_rows"] = training_rows
    return manifest


def write_model_manifest(
    model_path: str | Path,
    backend: str,
    training_rows: int | None = None,
) -> Path:
    path = model_manifest_path(model_path)
    path.write_text(
        json.dumps(model_manifest(backend=backend, training_rows=training_rows), indent=2)
        + "\n",
        encoding="utf-8",
    )
    return path


def validate_model_manifest(model_path: str | Path, backend: str) -> None:
    path = model_manifest_path(model_path)
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifact_hash = payload.get("feature_schema_hash")
    expected_hash = feature_schema_hash()
    if artifact_hash != expected_hash:
        raise RuntimeError(
            f"{backend} model feature schema mismatch: artifact={artifact_hash} "
            f"expected={expected_hash}"
        )


@dataclass(frozen=True)
class ModelConfig:
    target_latency_ms: float = 25.0
    base_lap_time_s: float = 90.0
    pit_loss_s: float = 21.0


@dataclass(frozen=True)
class _PhysicsPrior:
    tire_wear_pct: float
    remaining_tire_life_laps: float
    grip_loss_pct: float
    overheating_probability: float
    cliff_probability: float
    brake_temp_next_lap_c: float
    lap_delta_s: float
    uncertainty_s: float


class _OnlineResidualLearner:
    """Small online ridge learner used as one member of the serving ensemble."""

    def __init__(self, feature_count: int, learning_rate: float, l2: float) -> None:
        self.learning_rate = learning_rate
        self.l2 = l2
        self.weights = [0.0] * feature_count

    def predict(self, features: list[float]) -> float:
        return sum(weight * value for weight, value in zip(self.weights, features, strict=True))

    def update(self, features: list[float], target: float) -> float:
        prediction = self.predict(features)
        error = target - prediction
        scale = 1.0 / sqrt(sum(value * value for value in features) + 1.0)
        for index, value in enumerate(features):
            gradient = error * value - self.l2 * self.weights[index]
            self.weights[index] += self.learning_rate * scale * gradient
        return error


class ServingModel(Protocol):
    backend_name: str

    def observe(self, features: OnlineFeatures) -> None: ...

    def predict(self, features: OnlineFeatures) -> Prediction: ...


def features_to_vector(features: OnlineFeatures) -> list[float]:
    compound_flags = [1.0 if features.compound == compound else 0.0 for compound in TireCompound]
    return [
        1.0,
        CIRCUIT_ENCODING.get(features.circuit, 0.0),
        features.rolling_lap_time_s / 130.0,
        features.tire_age_laps / 45.0,
        features.mean_tire_temp / 125.0,
        features.tire_temp_gradient / 35.0,
        features.brake_heat_index / 850.0,
        features.driver_aggression / 1.5,
        features.steering_entropy / 3.0,
        features.cumulative_tire_load / 25.0,
        features.degradation_acceleration / 2.0,
        features.ers_efficiency / 1.4,
        features.ers_soc,
        features.track_temp_c / 65.0,
        features.fuel_kg / 110.0,
        features.dirty_air_risk,
        features.circuit_base_lap_s / 130.0,
        *compound_flags,
    ]


class HybridOnlineEnsembleModel:
    """Physics-informed online ensemble for low-latency race strategy inference.

    The model uses Formula One domain priors for tire and thermal behavior, then
    learns session-specific pace residuals online from observed lap times. The
    ensemble keeps several ridge learners with different learning rates and
    regularization strengths; their disagreement is used as a cheap uncertainty
    estimate alongside a conformal residual buffer.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.backend_name = "hybrid-online-ensemble"
        feature_count = len(features_to_vector(self._cold_start_features()))
        self.learners = [
            _OnlineResidualLearner(feature_count, learning_rate=0.030, l2=0.002),
            _OnlineResidualLearner(feature_count, learning_rate=0.018, l2=0.006),
            _OnlineResidualLearner(feature_count, learning_rate=0.010, l2=0.012),
            _OnlineResidualLearner(feature_count, learning_rate=0.006, l2=0.020),
        ]
        self.observations = 0
        self.absolute_errors: deque[float] = deque(maxlen=250)

    def observe(self, features: OnlineFeatures) -> None:
        target_delta = features.rolling_lap_time_s - self.config.base_lap_time_s
        if abs(target_delta) > 20.0:
            return

        prior = self._physics_prior(features)
        target_residual = self._clamp(target_delta - prior.lap_delta_s, -4.0, 4.0)
        vector = features_to_vector(features)
        errors = [learner.update(vector, target_residual) for learner in self.learners]
        self.absolute_errors.append(abs(mean(errors)))
        self.observations += 1

    def predict(self, features: OnlineFeatures) -> Prediction:
        prior = self._physics_prior(features)
        vector = features_to_vector(features)
        residuals = [learner.predict(vector) for learner in self.learners]
        residual = mean(residuals)
        warmup = min(1.0, self.observations / 18.0)
        residual = self._clamp(residual * warmup, -2.5, 2.5)

        lap_delta = prior.lap_delta_s + residual
        if self.observations > 0:
            observed_delta = features.rolling_lap_time_s - self.config.base_lap_time_s
            anchor_weight = min(0.92, 0.72 + self.observations / 120.0)
            lap_delta = observed_delta * anchor_weight + lap_delta * (1.0 - anchor_weight)
        if len(residuals) > 1:
            m = mean(residuals)
            ensemble_spread = (sum((x - m) ** 2 for x in residuals) / len(residuals)) ** 0.5
        else:
            ensemble_spread = 0.0
        conformal_error = self._conformal_error()
        anchor_disagreement = 0.0
        if self.observations > 0:
            anchor_disagreement = abs(
                lap_delta - (features.rolling_lap_time_s - self.config.base_lap_time_s)
            )
        uncertainty = max(
            0.12,
            prior.uncertainty_s
            + 1.35 * ensemble_spread
            + conformal_error * warmup
            + anchor_disagreement * 0.35,
        )

        learned_stress = max(0.0, residual) * 0.7 + ensemble_spread * 0.4
        grip_loss = min(45.0, prior.grip_loss_pct + learned_stress)
        cliff = min(0.995, prior.cliff_probability + self._sigmoid(residual - 1.25) * 0.035)

        return Prediction(
            session_id=features.session_id,
            car_id=features.car_id,
            lap=features.lap,
            tire_wear_pct=prior.tire_wear_pct,
            remaining_tire_life_laps=prior.remaining_tire_life_laps,
            grip_loss_pct=grip_loss,
            overheating_probability=prior.overheating_probability,
            cliff_probability=cliff,
            brake_temp_next_lap_c=prior.brake_temp_next_lap_c,
            ers_efficiency=features.ers_efficiency,
            next_lap_delta_s=lap_delta,
            uncertainty_low_s=lap_delta - uncertainty,
            uncertainty_high_s=lap_delta + uncertainty,
        )

    def prediction_from_lap_delta(
        self,
        features: OnlineFeatures,
        lap_delta: float,
        uncertainty_extra: float = 0.0,
    ) -> Prediction:
        prior = self._physics_prior(features)
        delta_gap = abs(lap_delta - prior.lap_delta_s)
        uncertainty = max(0.14, prior.uncertainty_s + uncertainty_extra + delta_gap * 0.25)
        grip_adjustment = max(0.0, lap_delta - prior.lap_delta_s) * 0.6

        return Prediction(
            session_id=features.session_id,
            car_id=features.car_id,
            lap=features.lap,
            tire_wear_pct=prior.tire_wear_pct,
            remaining_tire_life_laps=prior.remaining_tire_life_laps,
            grip_loss_pct=min(45.0, prior.grip_loss_pct + grip_adjustment),
            overheating_probability=prior.overheating_probability,
            cliff_probability=prior.cliff_probability,
            brake_temp_next_lap_c=prior.brake_temp_next_lap_c,
            ers_efficiency=features.ers_efficiency,
            next_lap_delta_s=lap_delta,
            uncertainty_low_s=lap_delta - uncertainty,
            uncertainty_high_s=lap_delta + uncertainty,
        )

    def _physics_prior(self, features: OnlineFeatures) -> _PhysicsPrior:
        life = COMPOUND_LIFE[features.compound]
        thermal_penalty = max(0.0, features.mean_tire_temp - 102.0) * 0.018
        track_penalty = max(0.0, features.track_temp_c - 38.0) * 0.010
        aggression_penalty = features.driver_aggression * 0.42
        slip_penalty = features.degradation_acceleration * 2.3
        wear_rate = 1.0 + thermal_penalty + track_penalty + aggression_penalty + slip_penalty
        load_adjustment = max(0.0, features.cumulative_tire_load)
        effective_age = features.tire_age_laps + load_adjustment * 0.12
        tire_wear_pct = min(100.0, effective_age / life * 100.0)
        remaining = max(0.0, life - effective_age)

        overheating = self._sigmoid((features.mean_tire_temp - 106.0) / 3.8)
        cliff = self._sigmoid(
            (tire_wear_pct - 78.0) / 7.5 + features.degradation_acceleration * 1.2
        )
        grip_loss = min(45.0, tire_wear_pct * 0.28 * wear_rate + overheating * 8.0)
        brake_next = (
            features.brake_heat_index * 0.018
            + features.mean_tire_temp * 0.35
            + features.track_temp_c * 1.4
        )
        brake_next = max(300.0, min(1200.0, brake_next))

        lap_delta = (
            COMPOUND_PACE[features.compound]
            + grip_loss * 0.035
            + overheating * 0.32
            + features.dirty_air_risk * 0.45
            + max(0.0, 35.0 - features.fuel_kg) * -0.018
            + max(0.0, 0.35 - features.ers_soc) * 0.75
        )
        uncertainty = 0.16 + cliff * 0.42 + overheating * 0.20

        return _PhysicsPrior(
            tire_wear_pct=tire_wear_pct,
            remaining_tire_life_laps=remaining,
            grip_loss_pct=grip_loss,
            overheating_probability=overheating,
            cliff_probability=cliff,
            brake_temp_next_lap_c=brake_next,
            lap_delta_s=lap_delta,
            uncertainty_s=uncertainty,
        )

    def _conformal_error(self) -> float:
        if len(self.absolute_errors) < 8:
            return 0.0
        ordered = sorted(self.absolute_errors)
        index = min(len(ordered) - 1, int(len(ordered) * 0.80))
        return ordered[index]

    @staticmethod
    def _cold_start_features() -> OnlineFeatures:
        return OnlineFeatures(
            session_id="cold-start",
            car_id="car",
            lap=1,
            compound=TireCompound.MEDIUM,
            circuit="synthetic",
            tire_age_laps=0,
            mean_tire_temp=95.0,
            tire_temp_gradient=2.0,
            brake_heat_index=350.0,
            driver_aggression=0.5,
            steering_entropy=1.0,
            cumulative_tire_load=0.0,
            degradation_acceleration=0.0,
            ers_efficiency=1.0,
            ers_soc=0.8,
            track_temp_c=35.0,
            fuel_kg=95.0,
            rolling_lap_time_s=90.0,
            dirty_air_risk=0.0,
            circuit_base_lap_s=90.0,
        )

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + exp(-value))

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return min(high, max(low, value))


class XGBoostModel:
    """XGBoost artifact adapter with hybrid online fallback and residual guardrails."""

    def __init__(
        self,
        model_path: str | Path,
        config: ModelConfig | None = None,
        fallback: HybridOnlineEnsembleModel | None = None,
    ) -> None:
        self.config = config or ModelConfig()
        self.model_path = Path(model_path)
        self.fallback = fallback or HybridOnlineEnsembleModel(self.config)
        self.backend_name = "xgboost"
        try:
            import xgboost as xgb
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("XGBoost backend requested but xgboost is not installed") from exc
        if not self.model_path.exists():
            raise RuntimeError(f"XGBoost model artifact does not exist: {self.model_path}")
        self._xgb = xgb
        self._np = np
        self.booster = xgb.Booster()
        self.booster.load_model(str(self.model_path))
        validate_model_manifest(self.model_path, self.backend_name)
        artifact_hash = self.booster.attr("feature_schema_hash")
        if artifact_hash is not None and artifact_hash != feature_schema_hash():
            raise RuntimeError(
                f"XGBoost model feature schema mismatch: artifact={artifact_hash} "
                f"expected={feature_schema_hash()}"
            )
        self.booster.set_param({"device": "cpu", "nthread": 1})
        warmup = self._np.asarray(
            [features_to_vector(HybridOnlineEnsembleModel._cold_start_features())]
        )
        self.booster.inplace_predict(warmup)

    def observe(self, features: OnlineFeatures) -> None:
        self.fallback.observe(features)

    def predict(self, features: OnlineFeatures) -> Prediction:
        fallback = self.fallback.predict(features)
        vector = features_to_vector(features)
        matrix = self._np.asarray([vector], dtype="float32")
        raw_prediction = self.booster.inplace_predict(matrix)
        outputs = self._coerce_outputs(raw_prediction)

        lap_delta = self._clamp(outputs[0], -25.0, 15.0)
        tire_wear_pct = fallback.tire_wear_pct
        cliff_probability = fallback.cliff_probability
        brake_temp_next_lap_c = fallback.brake_temp_next_lap_c
        if len(outputs) >= 2:
            tire_wear_pct = self._clamp(outputs[1], 0.0, 100.0)
        if len(outputs) >= 3:
            cliff_probability = self._clamp(outputs[2], 0.0, 0.995)
        if len(outputs) >= 4:
            brake_temp_next_lap_c = self._clamp(outputs[3], 300.0, 1200.0)

        fallback_width = fallback.uncertainty_high_s - fallback.uncertainty_low_s
        model_delta_gap = abs(lap_delta - fallback.next_lap_delta_s)
        uncertainty = max(0.14, min(2.65, fallback_width / 2.0 + model_delta_gap * 0.15))
        grip_adjustment = max(0.0, lap_delta - fallback.next_lap_delta_s) * 0.6

        return Prediction(
            session_id=features.session_id,
            car_id=features.car_id,
            lap=features.lap,
            tire_wear_pct=tire_wear_pct,
            remaining_tire_life_laps=fallback.remaining_tire_life_laps,
            grip_loss_pct=min(45.0, fallback.grip_loss_pct + grip_adjustment),
            overheating_probability=fallback.overheating_probability,
            cliff_probability=cliff_probability,
            brake_temp_next_lap_c=brake_temp_next_lap_c,
            ers_efficiency=features.ers_efficiency,
            next_lap_delta_s=lap_delta,
            uncertainty_low_s=lap_delta - uncertainty,
            uncertainty_high_s=lap_delta + uncertainty,
        )

    @staticmethod
    def _coerce_outputs(raw_prediction: object) -> list[float]:
        if hasattr(raw_prediction, "tolist"):
            raw_prediction = raw_prediction.tolist()
        if (
            isinstance(raw_prediction, list)
            and raw_prediction
            and isinstance(raw_prediction[0], list)
        ):
            return [float(value) for value in raw_prediction[0]]
        if isinstance(raw_prediction, list):
            return [float(raw_prediction[0])]
        return [float(raw_prediction)]

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return min(high, max(low, value))


class LightGBMModel:
    """LightGBM artifact adapter with hybrid online fallback."""

    def __init__(
        self,
        model_path: str | Path,
        config: ModelConfig | None = None,
        fallback: HybridOnlineEnsembleModel | None = None,
    ) -> None:
        self.config = config or ModelConfig()
        self.model_path = Path(model_path)
        self.fallback = fallback or HybridOnlineEnsembleModel(self.config)
        self.backend_name = "lightgbm"
        try:
            import lightgbm as lgb
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("LightGBM backend requested but lightgbm is not installed") from exc
        if not self.model_path.exists():
            raise RuntimeError(f"LightGBM model artifact does not exist: {self.model_path}")
        self._np = np
        self.booster = lgb.Booster(model_file=str(self.model_path))
        validate_model_manifest(self.model_path, self.backend_name)
        self.booster.predict(
            self._np.asarray(
                [features_to_vector(HybridOnlineEnsembleModel._cold_start_features())]
            )
        )

    def observe(self, features: OnlineFeatures) -> None:
        self.fallback.observe(features)

    def predict(self, features: OnlineFeatures) -> Prediction:
        vector = self._np.asarray([features_to_vector(features)], dtype="float32")
        raw_prediction = self.booster.predict(vector)
        lap_delta = _clamp(_first_prediction(raw_prediction), -25.0, 15.0)
        fallback = self.fallback.predict(features)
        width = (fallback.uncertainty_high_s - fallback.uncertainty_low_s) / 2.0
        return self.fallback.prediction_from_lap_delta(
            features, lap_delta, uncertainty_extra=width * 0.2
        )


class CatBoostModel:
    """CatBoost artifact adapter with hybrid online fallback."""

    def __init__(
        self,
        model_path: str | Path,
        config: ModelConfig | None = None,
        fallback: HybridOnlineEnsembleModel | None = None,
    ) -> None:
        self.config = config or ModelConfig()
        self.model_path = Path(model_path)
        self.fallback = fallback or HybridOnlineEnsembleModel(self.config)
        self.backend_name = "catboost"
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise RuntimeError("CatBoost backend requested but catboost is not installed") from exc
        if not self.model_path.exists():
            raise RuntimeError(f"CatBoost model artifact does not exist: {self.model_path}")
        self.model = CatBoostRegressor()
        self.model.load_model(str(self.model_path))
        validate_model_manifest(self.model_path, self.backend_name)
        self.model.predict([features_to_vector(HybridOnlineEnsembleModel._cold_start_features())])

    def observe(self, features: OnlineFeatures) -> None:
        self.fallback.observe(features)

    def predict(self, features: OnlineFeatures) -> Prediction:
        raw_prediction = self.model.predict([features_to_vector(features)])
        lap_delta = _clamp(_first_prediction(raw_prediction), -25.0, 15.0)
        fallback = self.fallback.predict(features)
        width = (fallback.uncertainty_high_s - fallback.uncertainty_low_s) / 2.0
        return self.fallback.prediction_from_lap_delta(
            features, lap_delta, uncertainty_extra=width * 0.2
        )


class SequenceTorchModel:
    """TorchScript adapter for LSTM/TFT-style sequence models.

    The artifact is expected to accept a 2-D float tensor with the shared online
    feature vector. Sequence assembly can be moved into the feature store later
    without changing the InferenceEngine contract.
    """

    def __init__(
        self,
        model_path: str | Path,
        config: ModelConfig | None = None,
        fallback: HybridOnlineEnsembleModel | None = None,
    ) -> None:
        self.config = config or ModelConfig()
        self.model_path = Path(model_path)
        self.fallback = fallback or HybridOnlineEnsembleModel(self.config)
        self.backend_name = "sequence-torch"
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("LSTM/TFT backend requested but torch is not installed") from exc
        if not self.model_path.exists():
            raise RuntimeError(f"Sequence model artifact does not exist: {self.model_path}")
        self._torch = torch
        self.model = torch.jit.load(str(self.model_path), map_location="cpu")
        validate_model_manifest(self.model_path, self.backend_name)
        self.model.eval()
        warmup = torch.tensor(
            [features_to_vector(HybridOnlineEnsembleModel._cold_start_features())]
        )
        with torch.no_grad():
            self.model(warmup.float())

    def observe(self, features: OnlineFeatures) -> None:
        self.fallback.observe(features)

    def predict(self, features: OnlineFeatures) -> Prediction:
        vector = self._torch.tensor([features_to_vector(features)], dtype=self._torch.float32)
        with self._torch.no_grad():
            raw_prediction = self.model(vector)
        lap_delta = _clamp(_first_prediction(raw_prediction), -25.0, 15.0)
        return self.fallback.prediction_from_lap_delta(features, lap_delta, uncertainty_extra=0.18)


class KalmanFilterModel:
    """Dependency-free Kalman filter backend for online pace smoothing."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.fallback = HybridOnlineEnsembleModel(self.config)
        self.backend_name = "kalman"
        self.state_by_car: dict[tuple[str, str], tuple[float, float]] = {}
        self.process_variance = 0.018
        self.measurement_variance = 0.12

    def observe(self, features: OnlineFeatures) -> None:
        self.fallback.observe(features)
        if features.rolling_lap_time_s <= 0:
            return
        key = (features.session_id, features.car_id)
        measurement = _clamp(features.rolling_lap_time_s - self.config.base_lap_time_s, -25.0, 15.0)
        estimate, variance = self.state_by_car.get(key, (measurement, 1.0))
        predicted_variance = variance + self.process_variance
        gain = predicted_variance / (predicted_variance + self.measurement_variance)
        estimate = estimate + gain * (measurement - estimate)
        variance = (1.0 - gain) * predicted_variance
        self.state_by_car[key] = (estimate, variance)

    def predict(self, features: OnlineFeatures) -> Prediction:
        fallback = self.fallback.predict(features)
        estimate, variance = self.state_by_car.get(
            (features.session_id, features.car_id),
            (fallback.next_lap_delta_s, 0.25),
        )
        blended = 0.62 * estimate + 0.38 * fallback.next_lap_delta_s
        return self.fallback.prediction_from_lap_delta(
            features,
            _clamp(blended, -25.0, 15.0),
            uncertainty_extra=sqrt(max(variance, 0.0)),
        )


class RiverOnlineModel:
    """River online regressor backend with hybrid fallback."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.fallback = HybridOnlineEnsembleModel(self.config)
        self.backend_name = "river-online"
        try:
            from river import compose, linear_model, preprocessing
        except ImportError as exc:
            raise RuntimeError("River backend requested but river is not installed") from exc
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LinearRegression(),
        )
        self.observations = 0

    def observe(self, features: OnlineFeatures) -> None:
        self.fallback.observe(features)
        target = features.rolling_lap_time_s - self.config.base_lap_time_s
        if abs(target) > 20.0:
            return
        self.model.learn_one(self._features(features), target)
        self.observations += 1

    def predict(self, features: OnlineFeatures) -> Prediction:
        fallback = self.fallback.predict(features)
        if self.observations < 6:
            return fallback
        prediction = self.model.predict_one(self._features(features))
        if prediction is None:
            return fallback
        warmup = min(1.0, self.observations / 24.0)
        blended = (1.0 - warmup) * fallback.next_lap_delta_s + warmup * float(prediction)
        return self.fallback.prediction_from_lap_delta(features, _clamp(blended, -25.0, 15.0))

    @staticmethod
    def _features(features: OnlineFeatures) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, features_to_vector(features), strict=True))


def _first_prediction(raw_prediction: object) -> float:
    if hasattr(raw_prediction, "detach"):
        raw_prediction = raw_prediction.detach().cpu().reshape(-1).tolist()
    elif hasattr(raw_prediction, "tolist"):
        raw_prediction = raw_prediction.tolist()
    while (
        isinstance(raw_prediction, list)
        and raw_prediction
        and isinstance(raw_prediction[0], list)
    ):
        raw_prediction = raw_prediction[0]
    if isinstance(raw_prediction, list):
        return float(raw_prediction[0])
    return float(raw_prediction)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def create_serving_model(
    config: ModelConfig,
    backend: str = "auto",
    xgboost_model_path: str = "models/xgboost_lap_delta.json",
    lightgbm_model_path: str = "models/lightgbm_lap_delta.txt",
    catboost_model_path: str = "models/catboost_lap_delta.cbm",
    sequence_model_path: str = "models/sequence_lap_delta.pt",
    model_artifact_id: str = "",
    model_artifact_root: str = "artifacts/models",
) -> ServingModel:
    normalized_backend = backend.strip().lower()
    if model_artifact_id.strip():
        from f1_strategy.artifacts import resolve_model_artifact

        resolved_artifact_id = model_artifact_id.strip()
        artifact_backend, artifact_model_path = resolve_model_artifact(
            resolved_artifact_id,
            artifact_root=model_artifact_root,
        )
        normalized_backend = artifact_backend.strip().lower()
        if normalized_backend == "xgboost":
            xgboost_model_path = str(artifact_model_path)
        elif normalized_backend == "lightgbm":
            lightgbm_model_path = str(artifact_model_path)
        elif normalized_backend == "catboost":
            catboost_model_path = str(artifact_model_path)
        elif normalized_backend in {"sequence", "sequence-torch", "lstm", "tft"}:
            sequence_model_path = str(artifact_model_path)
        else:
            raise ValueError(f"Unsupported artifact model backend: {artifact_backend}")
    if normalized_backend in {"hybrid", "hybrid-online-ensemble"}:
        return HybridOnlineEnsembleModel(config)
    if normalized_backend == "kalman":
        return KalmanFilterModel(config)
    if normalized_backend in {"river", "river-online"}:
        return RiverOnlineModel(config)

    artifact_backends = {
        "xgboost": lambda: XGBoostModel(xgboost_model_path, config=config),
        "lightgbm": lambda: LightGBMModel(lightgbm_model_path, config=config),
        "catboost": lambda: CatBoostModel(catboost_model_path, config=config),
        "lstm": lambda: SequenceTorchModel(sequence_model_path, config=config),
        "tft": lambda: SequenceTorchModel(sequence_model_path, config=config),
        "sequence": lambda: SequenceTorchModel(sequence_model_path, config=config),
        "sequence-torch": lambda: SequenceTorchModel(sequence_model_path, config=config),
    }
    if normalized_backend in artifact_backends:
        model = artifact_backends[normalized_backend]()
        if model_artifact_id.strip():
            setattr(model, "artifact_id", model_artifact_id.strip())
        return model
    if normalized_backend != "auto":
        raise ValueError(f"Unsupported model backend: {backend}")

    for loader in (
        artifact_backends["xgboost"],
        artifact_backends["lightgbm"],
        artifact_backends["catboost"],
        artifact_backends["sequence"],
    ):
        try:
            model = loader()
            if model_artifact_id.strip():
                setattr(model, "artifact_id", model_artifact_id.strip())
            return model
        except RuntimeError:
            continue
    return HybridOnlineEnsembleModel(config)


BaselineProbabilisticModel = HybridOnlineEnsembleModel
