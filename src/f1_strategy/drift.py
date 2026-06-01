from __future__ import annotations

from statistics import mean, pstdev

from f1_strategy.domain import DriftReport, OnlineFeatures


class DriftDetector:
    def __init__(self, threshold_z: float = 3.0) -> None:
        self.threshold_z = threshold_z
        self._baseline: dict[str, tuple[float, float]] = {}

    def fit_baseline(self, features: list[OnlineFeatures]) -> None:
        self._baseline = {}
        for name in self._feature_names():
            values = [float(getattr(row, name)) for row in features]
            if values:
                self._baseline[name] = (mean(values), max(pstdev(values), 1e-6))

    def detect(self, feature: OnlineFeatures) -> DriftReport:
        scores: dict[str, float] = {}
        alerts: list[str] = []
        for name, (mu, sigma) in self._baseline.items():
            value = float(getattr(feature, name))
            score = abs(value - mu) / sigma
            scores[name] = score
            if score >= self.threshold_z:
                alerts.append(f"{name} drift z={score:.2f}")
        return DriftReport(drifted=bool(alerts), feature_scores=scores, alerts=alerts)

    @staticmethod
    def _feature_names() -> list[str]:
        return [
            "mean_tire_temp",
            "tire_temp_gradient",
            "brake_heat_index",
            "driver_aggression",
            "cumulative_tire_load",
            "degradation_acceleration",
            "ers_efficiency",
            "track_temp_c",
            "dirty_air_risk",
        ]
