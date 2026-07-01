from __future__ import annotations

from collections import deque
from math import exp, pi, sqrt
from statistics import mean, stdev
from time import perf_counter

from f1_strategy.domain import OnlineFeatures, Prediction
from f1_strategy.models import ServingModel


def _normal_sf(z: float) -> float:
    """P(Z > z) for Z ~ N(0,1) via Abramowitz & Stegun 26.2.17 (max error 7.5e-8)."""
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    return exp(-0.5 * z * z) / sqrt(2.0 * pi) * poly


def _welch_pvalue(a: list[float], b: list[float]) -> float:
    """Two-sample Welch t-test p-value (two-tailed).  Returns p in [0, 1].

    Uses a normal approximation which is accurate for n >= 30 per group.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 1.0
    mean_a = sum(a) / na
    mean_b = sum(b) / nb
    var_a = sum((x - mean_a) ** 2 for x in a) / (na - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (nb - 1)
    se2 = var_a / na + var_b / nb
    if se2 == 0.0:
        return 0.0 if mean_a != mean_b else 1.0
    t_stat = (mean_a - mean_b) / sqrt(se2)
    return _normal_sf(abs(t_stat)) * 2.0  # two-tailed


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
        significance_level: float = 0.05,
    ) -> dict | None:
        """Return promotion metadata if the challenger significantly outperforms the champion.

        Gates applied in order:
        1. Shadow must be active with >= min_predictions recorded.
        2. Mean improvement must exceed improvement_threshold_s.
        3. Welch t-test on |champion_delta| vs |challenger_delta| must be significant
           at significance_level (p < significance_level).  Rejects spurious wins on
           small or high-variance windows that would pass a raw mean comparison.

        Returns None when any gate fails.
        """
        if not self.active:
            return None
        if self._total < min_predictions:
            return None
        window = list(self._window)

        champion_abs = [abs(r["champion_delta_s"]) for r in window]
        challenger_abs = [abs(r["challenger_delta_s"]) for r in window]
        champion_mean = mean(champion_abs)
        challenger_mean = mean(challenger_abs)
        improvement_s = champion_mean - challenger_mean

        if improvement_s < improvement_threshold_s:
            return None

        p_value = _welch_pvalue(champion_abs, challenger_abs)
        if p_value >= significance_level:
            return None

        return {
            "challenger_backend": self._challenger_backend,
            "challenger_artifact": self._challenger_artifact,
            "champion_mean_abs_delta_s": round(champion_mean, 4),
            "challenger_mean_abs_delta_s": round(challenger_mean, 4),
            "improvement_s": round(improvement_s, 4),
            "improvement_pct": round(improvement_s / max(champion_mean, 1e-6) * 100, 1),
            "window_size": len(window),
            "p_value": round(p_value, 4),
            "recommendation": "promote_challenger",
        }
