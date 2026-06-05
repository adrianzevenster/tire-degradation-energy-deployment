from __future__ import annotations

from collections import deque
from statistics import mean, stdev
from time import perf_counter

from f1_strategy.domain import OnlineFeatures, Prediction
from f1_strategy.models import ServingModel


class ShadowDeploymentManager:
    """Runs a challenger model in shadow alongside the champion.

    The challenger sees every observation and makes predictions but its
    output never reaches the caller — the champion always wins. Divergence
    stats accumulate in a rolling window and are exposed for comparison.

    promotion_candidate() surfaces the challenger as a promotion candidate
    when it consistently outperforms the champion on rolling lap-delta MAE
    with sufficient sample coverage.
    """

    _SIGNIFICANT_DELTA_S = 0.05
    _PROMOTE_MIN_PREDICTIONS = 50
    _PROMOTE_IMPROVEMENT_S = 0.02  # challenger must beat champion by ≥20ms

    def __init__(self, window_size: int = 200) -> None:
        self._challenger: ServingModel | None = None
        self._challenger_backend: str = "none"
        self._challenger_artifact: str = "unregistered"
        self._window: deque[dict] = deque(maxlen=window_size)
        self._total = 0
        self._diverged = 0

    def configure(
        self,
        model: ServingModel,
        backend: str,
        artifact_id: str = "unregistered",
    ) -> None:
        self._challenger = model
        self._challenger_backend = backend
        self._challenger_artifact = artifact_id
        self._window.clear()
        self._total = 0
        self._diverged = 0

    def disable(self) -> None:
        self._challenger = None
        self._challenger_backend = "none"
        self._challenger_artifact = "unregistered"
        self._window.clear()
        self._total = 0
        self._diverged = 0

    @property
    def active(self) -> bool:
        return self._challenger is not None

    def observe(self, features: OnlineFeatures, champion: Prediction) -> None:
        """Run challenger inference and record divergence against champion."""
        if self._challenger is None:
            return
        try:
            start = perf_counter()
            self._challenger.observe(features)
            challenger_pred: Prediction = self._challenger.predict(features)
            latency_ms = (perf_counter() - start) * 1000.0
        except Exception:
            return

        delta = champion.next_lap_delta_s - challenger_pred.next_lap_delta_s
        abs_delta = abs(delta)
        self._total += 1
        if abs_delta >= self._SIGNIFICANT_DELTA_S:
            self._diverged += 1

        self._window.append(
            {
                "lap": champion.lap,
                "champion_delta_s": champion.next_lap_delta_s,
                "challenger_delta_s": challenger_pred.next_lap_delta_s,
                "delta_s": delta,
                "abs_delta_s": abs_delta,
                "champion_wear": champion.tire_wear_pct,
                "challenger_wear": challenger_pred.tire_wear_pct,
                "champion_cliff": champion.cliff_probability,
                "challenger_cliff": challenger_pred.cliff_probability,
                "challenger_latency_ms": latency_ms,
            }
        )

    def status(self) -> dict:
        window = list(self._window)
        deltas = [r["abs_delta_s"] for r in window]
        divergence_rate = self._diverged / max(self._total, 1)
        return {
            "active": self.active,
            "challenger_backend": self._challenger_backend,
            "challenger_artifact": self._challenger_artifact,
            "total_predictions": self._total,
            "diverged_predictions": self._diverged,
            "divergence_rate": divergence_rate,
            "mean_abs_delta_s": mean(deltas) if deltas else 0.0,
            "max_abs_delta_s": max(deltas) if deltas else 0.0,
            "std_delta_s": stdev(deltas) if len(deltas) >= 2 else 0.0,
            "window_size": len(window),
            "recent": window[-40:],
            "promotion_candidate": self.promotion_candidate(),
        }

    def promotion_candidate(
        self,
        min_predictions: int = _PROMOTE_MIN_PREDICTIONS,
        improvement_threshold_s: float = _PROMOTE_IMPROVEMENT_S,
    ) -> dict | None:
        """Return promotion metadata if the challenger consistently outperforms.

        Returns None when:
        - Shadow is inactive
        - Fewer than min_predictions have been recorded
        - Challenger is not better than champion by improvement_threshold_s

        The comparison is on mean absolute lap-delta (a proxy for MAE since
        both models see the same inputs but we have no ground-truth labels
        during shadow serving).
        """
        if not self.active:
            return None
        if self._total < min_predictions:
            return None
        window = list(self._window)

        champion_mean = mean(abs(r["champion_delta_s"]) for r in window)
        challenger_mean = mean(abs(r["challenger_delta_s"]) for r in window)
        improvement_s = champion_mean - challenger_mean

        if improvement_s < improvement_threshold_s:
            return None

        return {
            "challenger_backend": self._challenger_backend,
            "challenger_artifact": self._challenger_artifact,
            "champion_mean_abs_delta_s": round(champion_mean, 4),
            "challenger_mean_abs_delta_s": round(challenger_mean, 4),
            "improvement_s": round(improvement_s, 4),
            "improvement_pct": round(improvement_s / max(champion_mean, 1e-6) * 100, 1),
            "window_size": len(window),
            "recommendation": "promote_challenger",
        }
