from __future__ import annotations

from collections import defaultdict, deque
from math import log
from statistics import mean

from f1_strategy.domain import DriftReport, OnlineFeatures


def _pstdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5


class DriftDetector:
    """Z-score + PSI + concept drift detector.

    Z-score flags individual point anomalies against a fitted baseline.
    PSI (Population Stability Index) detects distributional shift over a
    rolling window of recent observations vs the baseline distribution.
    Concept drift tracks rolling prediction error growth beyond the
    baseline error rate recorded after the initial warm-up period.
    """

    def __init__(
        self,
        threshold_z: float = 3.0,
        psi_warn: float = 0.10,
        psi_alert: float = 0.20,
        window_size: int = 60,
    ) -> None:
        self.threshold_z = threshold_z
        self.psi_warn = psi_warn
        self.psi_alert = psi_alert
        self._window_size = window_size
        self._baseline: dict[str, tuple[float, float]] = {}
        self._baseline_samples: dict[str, list[float]] = {}
        self._recent: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._error_buffer: deque[float] = deque(maxlen=window_size)
        self._error_baseline: tuple[float, float] | None = None

    def fit_baseline(self, features: list[OnlineFeatures]) -> None:
        self._baseline = {}
        self._baseline_samples = {}
        for name in self._feature_names():
            values = [float(getattr(row, name)) for row in features]
            if values:
                self._baseline[name] = (mean(values), max(_pstdev(values), 1e-6))
                self._baseline_samples[name] = values

    def record_error(self, absolute_error: float) -> None:
        """Feed an absolute prediction residual for concept drift tracking."""
        self._error_buffer.append(abs(absolute_error))
        if (
            len(self._error_buffer) >= self._window_size // 2
            and self._error_baseline is None
        ):
            mu = mean(self._error_buffer)
            sigma = max(_pstdev(list(self._error_buffer)), 1e-6)
            self._error_baseline = (mu, sigma)

    def detect(self, feature: OnlineFeatures) -> DriftReport:
        scores: dict[str, float] = {}
        alerts: list[str] = []

        for name, (mu, sigma) in self._baseline.items():
            value = float(getattr(feature, name))
            self._recent[name].append(value)
            score = abs(value - mu) / sigma
            scores[name] = score
            if score >= self.threshold_z:
                alerts.append(f"{name} drift z={score:.2f}")

        for name, baseline_vals in self._baseline_samples.items():
            recent = list(self._recent[name])
            if len(recent) >= 10:
                psi = _psi(recent, baseline_vals)
                scores[f"psi_{name}"] = psi
                if psi >= self.psi_alert:
                    alerts.append(f"{name} PSI={psi:.3f}")
                elif psi >= self.psi_warn:
                    alerts.append(f"{name} PSI warn={psi:.3f}")

        if self._error_baseline is not None and len(self._error_buffer) >= 10:
            mu, sigma = self._error_baseline
            recent_mae = mean(self._error_buffer)
            error_z = (recent_mae - mu) / sigma
            scores["concept_drift_z"] = error_z
            if error_z >= self.threshold_z:
                alerts.append(
                    f"concept drift error_z={error_z:.2f} recent_mae={recent_mae:.3f}s"
                )

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


def _psi(actual: list[float], expected: list[float], bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    lo = min(min(actual), min(expected))
    hi = max(max(actual), max(expected))
    if hi == lo:
        return 0.0
    span = hi - lo

    def fractions(values: list[float]) -> list[float]:
        counts = [0] * bins
        for v in values:
            idx = min(bins - 1, int((v - lo) / span * bins))
            counts[idx] += 1
        total = len(values)
        return [max(c / total, 1e-4) for c in counts]

    actual_f = fractions(actual)
    expected_f = fractions(expected)
    return sum((a - e) * log(a / e) for a, e in zip(actual_f, expected_f))
