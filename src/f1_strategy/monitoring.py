from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import isfinite
from statistics import mean
from time import perf_counter

from f1_strategy.alerting import AlertPolicy, alerts_to_dicts, evaluate_alerts, health_score
from f1_strategy.domain import DriftReport, Prediction, StrategyRecommendation


ML_METRICS = [
    "rmse",
    "mae",
    "mape",
    "calibration_error",
    "prediction_interval_width_s",
    "model_mae_lap_delta_s",
    "model_rmse_lap_delta_s",
    "model_interval_coverage_pct",
    "model_evaluations_total",
    "model_alerts_total",
    "model_health_score",
    "tire_wear_pct",
    "tire_cliff_probability",
]

INFRA_METRICS = [
    "inference_latency_ms",
    "throughput_events_total",
    "predictions_total",
    "queue_depth",
    "drift_alerts_total",
    "deployment_ready",
]

RACING_METRICS = [
    "pit_stop_recommendations_total",
    "pit_stop_recommendation_accuracy",
    "tire_cliff_prediction_accuracy",
    "battery_depletion_error",
    "ers_efficiency",
    "next_lap_delta_s",
]


@dataclass
class MetricStore:
    counters: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    gauges: dict[str, float] = field(default_factory=dict)
    histograms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    labeled_counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = field(
        default_factory=lambda: defaultdict(float)
    )
    labeled_gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = field(
        default_factory=dict
    )

    def inc(self, name: str, amount: float = 1.0) -> None:
        self.counters[name] += amount

    def set(self, name: str, value: float) -> None:
        if isfinite(value):
            self.gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        if isfinite(value):
            self.histograms[name].append(value)

    def inc_labeled(self, name: str, labels: dict[str, str], amount: float = 1.0) -> None:
        self.labeled_counters[(name, _label_key(labels))] += amount

    def set_labeled(self, name: str, labels: dict[str, str], value: float) -> None:
        if isfinite(value):
            self.labeled_gauges[(name, _label_key(labels))] = value

    def render_prometheus(self) -> str:
        lines: list[str] = []
        for name, value in sorted(self.counters.items()):
            lines.append(f"# TYPE f1_{name} counter")
            lines.append(f"f1_{name} {value}")
        for (name, labels), value in sorted(self.labeled_counters.items()):
            lines.append(f"# TYPE f1_{name} counter")
            lines.append(f"f1_{name}{_render_labels(labels)} {value}")
        for name, value in sorted(self.gauges.items()):
            lines.append(f"# TYPE f1_{name} gauge")
            lines.append(f"f1_{name} {value}")
        for (name, labels), value in sorted(self.labeled_gauges.items()):
            lines.append(f"# TYPE f1_{name} gauge")
            lines.append(f"f1_{name}{_render_labels(labels)} {value}")
        for name, values in sorted(self.histograms.items()):
            if not values:
                continue
            ordered = sorted(values)
            lines.append(f"# TYPE f1_{name} summary")
            lines.append(f"f1_{name}_count {len(values)}")
            lines.append(f"f1_{name}_sum {sum(values)}")
            lines.append(f"f1_{name}_p50 {ordered[int((len(ordered) - 1) * 0.50)]}")
            lines.append(f"f1_{name}_p95 {ordered[int((len(ordered) - 1) * 0.95)]}")
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class EvaluationSample:
    absolute_error_s: float
    squared_error_s: float
    interval_width_s: float
    covered: bool
    cliff_correct: bool | None
    ending_soc_error: float | None


class MonitoringService:
    def __init__(
        self,
        store: MetricStore | None = None,
        window_size: int = 250,
        alert_policy: AlertPolicy | None = None,
    ) -> None:
        self.store = store or MetricStore()
        self._last_event_at = perf_counter()
        self.window_size = window_size
        self.alert_policy = alert_policy or AlertPolicy()
        self._model_evaluations: dict[tuple[str, str], deque[EvaluationSample]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._drift_counts: dict[tuple[str, str, str], int] = defaultdict(int)
        self._seen_alerts: set[tuple[str, str, str, str]] = set()

    def record_prediction(self, prediction: Prediction, latency_ms: float) -> None:
        now = perf_counter()
        elapsed = max(now - self._last_event_at, 1e-9)
        self._last_event_at = now
        self.store.inc("throughput_events_total")
        self.store.inc("predictions_total")
        self.store.set("throughput_events_per_second", 1.0 / elapsed)
        self.store.observe("inference_latency_ms", latency_ms)
        self.store.set("tire_wear_pct", prediction.tire_wear_pct)
        self.store.set("tire_cliff_probability", prediction.cliff_probability)
        self.store.set("overheating_probability", prediction.overheating_probability)
        self.store.set("brake_temp_next_lap_c", prediction.brake_temp_next_lap_c)
        self.store.set("ers_efficiency", prediction.ers_efficiency)
        self.store.set("next_lap_delta_s", prediction.next_lap_delta_s)
        self.store.set(
            "prediction_interval_width_s",
            prediction.uncertainty_high_s - prediction.uncertainty_low_s,
        )

    def record_strategy(self, recommendation: StrategyRecommendation) -> None:
        self.store.inc("pit_stop_recommendations_total")
        self.store.set(
            "undercut_success_probability",
            recommendation.pit_window.undercut_success_probability,
        )
        self.store.set(
            "safety_car_sensitivity",
            recommendation.pit_window.safety_car_sensitivity,
        )
        self.store.set("ers_ending_soc", recommendation.energy_plan.ending_soc)
        self.store.set("ers_expected_lap_gain_s", recommendation.energy_plan.expected_lap_gain_s)
        self.store.set("pace_target_delta_s", recommendation.pace_target_delta_s)

    def record_drift(
        self,
        report: DriftReport,
        model_backend: str = "unknown",
        model_artifact_id: str = "unregistered",
    ) -> None:
        if report.drifted:
            self.store.inc("drift_alerts_total", len(report.alerts))
            for alert in report.alerts:
                feature = alert.split(" ", 1)[0]
                self._drift_counts[(model_backend, model_artifact_id, feature)] += 1
        for name, score in report.feature_scores.items():
            self.store.set(f"drift_z_score_{name}", score)

    def record_queue_depth(self, depth: int) -> None:
        self.store.set("queue_depth", float(depth))

    def record_evaluation(
        self,
        actual_lap_delta_s: float,
        prediction: Prediction,
        actual_cliff: bool | None = None,
        actual_ending_soc: float | None = None,
    ) -> None:
        error = prediction.next_lap_delta_s - actual_lap_delta_s
        abs_error = abs(error)
        self.store.set("mae", abs_error)
        self.store.set("rmse", abs_error)
        if abs(actual_lap_delta_s) > 1e-6:
            self.store.set("mape", abs_error / abs(actual_lap_delta_s))
        covered = (
            prediction.uncertainty_low_s
            <= actual_lap_delta_s
            <= prediction.uncertainty_high_s
        )
        self.store.set("calibration_error", 0.0 if covered else 1.0)
        if actual_cliff is not None:
            predicted_cliff = prediction.cliff_probability >= 0.50
            self.store.set(
                "tire_cliff_prediction_accuracy",
                1.0 if predicted_cliff == actual_cliff else 0.0,
            )
        if actual_ending_soc is not None:
            self.store.set(
                "battery_depletion_error",
                abs(prediction.ers_efficiency - actual_ending_soc),
            )
        self._record_model_performance(
            prediction=prediction,
            abs_error=abs_error,
            squared_error=error * error,
            covered=covered,
            actual_cliff=actual_cliff,
            actual_ending_soc=actual_ending_soc,
        )

    def render_prometheus(self) -> str:
        return self.store.render_prometheus()

    def model_performance(self) -> list[dict]:
        summaries = []
        for (backend, artifact_id), samples in sorted(self._model_evaluations.items()):
            if not samples:
                continue
            summaries.append(_performance_summary(backend, artifact_id, list(samples)))
        return summaries

    def model_comparison(self) -> list[dict]:
        rows = []
        for item in self.model_performance():
            backend = str(item["backend"])
            artifact_id = str(item["artifact_id"])
            alerts = evaluate_alerts(
                performance=[item],
                drift_counts=dict(self._drift_counts),
                latency_p95_ms=0.0,
                current_backend=backend,
                current_artifact_id=artifact_id,
                policy=self.alert_policy,
            )
            row = dict(item)
            row["health_score"] = health_score(alerts)
            row["alert_count"] = len(alerts)
            row["critical_alert_count"] = sum(
                1 for alert in alerts if alert.severity == "critical"
            )
            rows.append(row)
        rows.sort(
            key=lambda item: (
                item["critical_alert_count"],
                item["mae_lap_delta_s"],
                -item["interval_coverage_pct"],
            )
        )
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        best_row = rows[0] if rows else None
        for row in rows:
            row["recommendation"] = _comparison_recommendation(row, best_row)
        return rows

    def model_alerts(
        self,
        current_backend: str,
        current_artifact_id: str,
        latency_p95_ms: float,
    ) -> dict:
        alerts = evaluate_alerts(
            performance=self.model_performance(),
            drift_counts=dict(self._drift_counts),
            latency_p95_ms=latency_p95_ms,
            current_backend=current_backend,
            current_artifact_id=current_artifact_id,
            policy=self.alert_policy,
        )
        score = health_score(alerts)
        self._record_alert_metrics(alerts, current_backend, current_artifact_id, score)
        return {
            "health_score": score,
            "alerts": alerts_to_dicts(alerts),
        }

    def record_deployment_readiness(
        self,
        backend: str,
        artifact_id: str,
        ready: bool,
    ) -> None:
        self.store.set_labeled(
            "deployment_ready",
            {"backend": backend, "artifact_id": artifact_id},
            1.0 if ready else 0.0,
        )

    def _record_model_performance(
        self,
        prediction: Prediction,
        abs_error: float,
        squared_error: float,
        covered: bool,
        actual_cliff: bool | None,
        actual_ending_soc: float | None,
    ) -> None:
        backend = prediction.model_backend or "unknown"
        artifact_id = prediction.model_artifact_id or "unregistered"
        predicted_cliff = prediction.cliff_probability >= 0.50
        sample = EvaluationSample(
            absolute_error_s=abs_error,
            squared_error_s=squared_error,
            interval_width_s=prediction.uncertainty_high_s - prediction.uncertainty_low_s,
            covered=covered,
            cliff_correct=(
                None if actual_cliff is None else predicted_cliff == actual_cliff
            ),
            ending_soc_error=(
                None
                if actual_ending_soc is None
                else abs(prediction.ers_efficiency - actual_ending_soc)
            ),
        )
        samples = self._model_evaluations[(backend, artifact_id)]
        samples.append(sample)
        summary = _performance_summary(backend, artifact_id, list(samples))
        labels = {"backend": backend, "artifact_id": artifact_id}
        self.store.inc_labeled("model_evaluations_total", labels)
        self.store.set_labeled("model_mae_lap_delta_s", labels, summary["mae_lap_delta_s"])
        self.store.set_labeled("model_rmse_lap_delta_s", labels, summary["rmse_lap_delta_s"])
        self.store.set_labeled(
            "model_interval_coverage_pct",
            labels,
            summary["interval_coverage_pct"],
        )
        self.store.set_labeled(
            "model_prediction_interval_width_s",
            labels,
            summary["mean_interval_width_s"],
        )
        if summary["cliff_accuracy_pct"] is not None:
            self.store.set_labeled(
                "model_cliff_accuracy_pct",
                labels,
                summary["cliff_accuracy_pct"],
            )
        if summary["mean_ending_soc_error"] is not None:
            self.store.set_labeled(
                "model_ending_soc_error",
                labels,
                summary["mean_ending_soc_error"],
            )

    def _record_alert_metrics(
        self,
        alerts: list,
        current_backend: str,
        current_artifact_id: str,
        score: float,
    ) -> None:
        self.store.set_labeled(
            "model_health_score",
            {"backend": current_backend, "artifact_id": current_artifact_id},
            score,
        )
        for alert in alerts:
            labels = {
                "backend": alert.backend,
                "artifact_id": alert.artifact_id,
                "severity": alert.severity,
                "type": alert.alert_type,
            }
            if alert.fingerprint not in self._seen_alerts:
                self.store.inc_labeled("model_alerts_total", labels)
                self._seen_alerts.add(alert.fingerprint)


def monitoring_catalog() -> dict[str, list[str]]:
    return {
        "ml": ML_METRICS,
        "infrastructure": INFRA_METRICS,
        "racing": RACING_METRICS,
    }


def _performance_summary(
    backend: str,
    artifact_id: str,
    samples: list[EvaluationSample],
) -> dict:
    count = len(samples)
    cliff_samples = [sample for sample in samples if sample.cliff_correct is not None]
    soc_samples = [sample for sample in samples if sample.ending_soc_error is not None]
    return {
        "backend": backend,
        "artifact_id": artifact_id,
        "window_size": count,
        "evaluations": count,
        "mae_lap_delta_s": mean(sample.absolute_error_s for sample in samples),
        "rmse_lap_delta_s": mean(sample.squared_error_s for sample in samples) ** 0.5,
        "interval_coverage_pct": (
            sum(1 for sample in samples if sample.covered) / count * 100.0
        ),
        "mean_interval_width_s": mean(sample.interval_width_s for sample in samples),
        "cliff_accuracy_pct": (
            None
            if not cliff_samples
            else sum(1 for sample in cliff_samples if sample.cliff_correct)
            / len(cliff_samples)
            * 100.0
        ),
        "mean_ending_soc_error": (
            None
            if not soc_samples
            else mean(
                sample.ending_soc_error
                for sample in soc_samples
                if sample.ending_soc_error is not None
            )
        ),
    }


def _comparison_recommendation(row: dict, best_row: dict | None) -> dict[str, str | float]:
    best_mae = float(best_row["mae_lap_delta_s"]) if best_row else float(row["mae_lap_delta_s"])
    mae = float(row["mae_lap_delta_s"])
    coverage = float(row["interval_coverage_pct"])
    health = float(row["health_score"])
    alert_count = int(row["alert_count"])
    critical_count = int(row["critical_alert_count"])
    mae_gap = mae - best_mae

    if (
        critical_count > 0
        or health < 55.0
        or coverage < 75.0
        or mae_gap > 0.08
    ):
        action = "retire"
        reason = "High error, weak coverage, or critical alerts"
    elif row.get("rank") == 1 and health >= 80.0 and coverage >= 90.0 and alert_count == 0:
        action = "promote"
        reason = "Best observed performer with strong coverage and no alerts"
    elif mae_gap <= 0.03 and health >= 70.0 and coverage >= 85.0 and critical_count == 0:
        action = "hold"
        reason = "Competitive but not clearly ahead of the field"
    else:
        action = "hold"
        reason = "Needs more evidence before promotion"

    return {
        "action": action,
        "reason": reason,
        "mae_gap_s": round(mae_gap, 6),
        "coverage_gap_pct": round(float(best_row["interval_coverage_pct"]) - coverage, 6) if best_row else 0.0,
        "health_score": round(health, 2),
    }


def _label_key(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, _escape_label_value(value)) for key, value in labels.items()))


def _render_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    payload = ",".join(f'{key}="{value}"' for key, value in labels)
    return "{" + payload + "}"


def _escape_label_value(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
