from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from f1_strategy.config import Settings, load_settings
from f1_strategy.deployment import deployment_readiness
from f1_strategy.domain import DriftReport, Prediction, StrategyRecommendation, TelemetryEvent
from f1_strategy.drift import DriftDetector
from f1_strategy.feature_store import OnlineFeatureStore
from f1_strategy.metadata import build_info
from f1_strategy.monitoring import MonitoringService
from f1_strategy.models import ModelConfig, ServingModel, create_serving_model, feature_schema_hash
from f1_strategy.optimization import StrategyOptimizer
from f1_strategy.persistence import PersistenceStore, create_persistence_store


class InferenceEngine:
    def __init__(
        self,
        feature_store: OnlineFeatureStore | None = None,
        model: ServingModel | None = None,
        optimizer: StrategyOptimizer | None = None,
        drift_detector: DriftDetector | None = None,
        monitoring: MonitoringService | None = None,
        persistence: PersistenceStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        config = ModelConfig(
            target_latency_ms=self.settings.target_latency_ms,
            base_lap_time_s=self.settings.base_lap_time_s,
            pit_loss_s=self.settings.pit_loss_s,
        )
        self.feature_store = feature_store or OnlineFeatureStore(
            window_size=self.settings.feature_window_size
        )
        self.model = model or create_serving_model(
            config=config,
            backend=self.settings.model_backend,
            xgboost_model_path=self.settings.xgboost_model_path,
            lightgbm_model_path=self.settings.lightgbm_model_path,
            catboost_model_path=self.settings.catboost_model_path,
            sequence_model_path=self.settings.sequence_model_path,
            model_artifact_id=self.settings.model_artifact_id,
            model_artifact_root=self.settings.model_artifact_root,
        )
        self.optimizer = optimizer or StrategyOptimizer(config)
        self.drift_detector = drift_detector or DriftDetector(self.settings.drift_threshold_z)
        self.monitoring = monitoring or MonitoringService()
        self.persistence = persistence or create_persistence_store(
            self.settings.persistence_backend,
            self.settings.duckdb_path,
        )
        self.latency_ms: list[float] = []
        self.build_info = build_info()

    def ingest(self, event: TelemetryEvent) -> Prediction:
        start = perf_counter()
        features = self.feature_store.ingest(event)
        self.model.observe(features)
        prediction = self._annotate_prediction(self.model.predict(features))
        latency = (perf_counter() - start) * 1000.0
        self.latency_ms.append(latency)
        self.monitoring.record_prediction(prediction, latency)
        self.persistence.record_telemetry(event)
        self.persistence.record_features(features)
        self.persistence.record_prediction(prediction, latency)
        if event.lap_time_s is not None:
            self._record_observed_lap(event, prediction)
        return prediction

    def predict(self, session_id: str, car_id: str) -> Prediction:
        features = self.feature_store.get(session_id, car_id)
        if features is None:
            raise KeyError(f"No online features for session={session_id} car={car_id}")
        start = perf_counter()
        prediction = self._annotate_prediction(self.model.predict(features))
        latency = (perf_counter() - start) * 1000.0
        self.latency_ms.append(latency)
        self.monitoring.record_prediction(prediction, latency)
        self.persistence.record_prediction(prediction, latency)
        return prediction

    def strategy(
        self, session_id: str, car_id: str, remaining_laps: int
    ) -> StrategyRecommendation:
        features = self.feature_store.get(session_id, car_id)
        if features is None:
            raise KeyError(f"No online features for session={session_id} car={car_id}")
        prediction = self._annotate_prediction(self.model.predict(features))
        recommendation = self.optimizer.recommend(features, prediction, remaining_laps)
        self.monitoring.record_strategy(recommendation)
        self.persistence.record_strategy(recommendation)
        return recommendation

    def record_evaluation(
        self,
        session_id: str,
        car_id: str,
        actual_lap_delta_s: float,
        actual_cliff: bool | None = None,
        actual_ending_soc: float | None = None,
    ) -> None:
        prediction = self.predict(session_id, car_id)
        self.monitoring.record_evaluation(
            actual_lap_delta_s=actual_lap_delta_s,
            prediction=prediction,
            actual_cliff=actual_cliff,
            actual_ending_soc=actual_ending_soc,
        )
        self.persistence.record_evaluation(
            session_id=session_id,
            car_id=car_id,
            actual_lap_delta_s=actual_lap_delta_s,
            prediction=prediction,
            actual_cliff=actual_cliff,
            actual_ending_soc=actual_ending_soc,
        )

    def drift(self, session_id: str, car_id: str) -> DriftReport:
        features = self.feature_store.get(session_id, car_id)
        if features is None:
            raise KeyError(f"No online features for session={session_id} car={car_id}")
        report = self.drift_detector.detect(features)
        self.monitoring.record_drift(
            report,
            model_backend=self._active_model_backend(),
            model_artifact_id=self._active_model_artifact_id(),
        )
        return report

    def latency_p95_ms(self) -> float:
        if not self.latency_ms:
            return 0.0
        ordered = sorted(self.latency_ms)
        index = min(len(ordered) - 1, int(len(ordered) * 0.95))
        return ordered[index]

    def model_performance(self) -> list[dict]:
        return self.monitoring.model_performance()

    def model_comparison(self) -> list[dict]:
        return self.monitoring.model_comparison()

    def model_alerts(self) -> dict:
        return self.monitoring.model_alerts(
            current_backend=self._active_model_backend(),
            current_artifact_id=self._active_model_artifact_id(),
            latency_p95_ms=self.latency_p95_ms(),
        )

    def deployment_readiness(self) -> dict:
        readiness = deployment_readiness(
            active_backend=self._active_model_backend(),
            active_artifact_id=self._active_model_artifact_id(),
            artifact_root=self.settings.model_artifact_root,
            alerts=self.model_alerts(),
            latency_p95_ms=self.latency_p95_ms(),
            target_latency_ms=self.settings.target_latency_ms,
            persistence_backend=getattr(self.persistence, "backend_name", "unknown"),
        )
        self.monitoring.record_deployment_readiness(
            readiness.active_backend,
            readiness.active_artifact_id,
            readiness.ready,
        )
        return readiness.to_dict()

    def _annotate_prediction(self, prediction: Prediction) -> Prediction:
        return replace(
            prediction,
            model_backend=self._active_model_backend(),
            model_artifact_id=self._active_model_artifact_id(),
            model_feature_schema_hash=feature_schema_hash(),
            app_version=self.build_info.version,
            build_sha=self.build_info.build_sha,
        )

    def _active_model_backend(self) -> str:
        return getattr(self.model, "backend_name", self.model.__class__.__name__)

    def _active_model_artifact_id(self) -> str:
        artifact_id = getattr(self.model, "artifact_id", "") or self.settings.model_artifact_id
        return artifact_id or "unregistered"

    def _record_observed_lap(self, event: TelemetryEvent, prediction: Prediction) -> None:
        actual_lap_delta_s = event.lap_time_s - self.settings.base_lap_time_s
        self.monitoring.record_evaluation(
            actual_lap_delta_s=actual_lap_delta_s,
            prediction=prediction,
        )
        self.persistence.record_evaluation(
            session_id=event.session_id,
            car_id=event.car_id,
            actual_lap_delta_s=actual_lap_delta_s,
            prediction=prediction,
        )
