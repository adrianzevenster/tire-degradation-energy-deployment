from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlertPolicy:
    max_mae_lap_delta_s: float = 1.25
    min_interval_coverage_pct: float = 50.0
    max_latency_p95_ms: float = 25.0
    min_evaluations: int = 1
    repeated_drift_threshold: int = 2


@dataclass(frozen=True)
class ModelAlert:
    alert_type: str
    severity: str
    backend: str
    artifact_id: str
    message: str
    value: float
    threshold: float

    @property
    def fingerprint(self) -> tuple[str, str, str, str]:
        return (self.backend, self.artifact_id, self.alert_type, self.severity)


def evaluate_alerts(
    performance: list[dict],
    drift_counts: dict[tuple[str, str, str], int],
    latency_p95_ms: float,
    current_backend: str,
    current_artifact_id: str,
    policy: AlertPolicy | None = None,
) -> list[ModelAlert]:
    selected_policy = policy or AlertPolicy()
    alerts: list[ModelAlert] = []
    active_performance = _active_performance(
        performance,
        backend=current_backend,
        artifact_id=current_artifact_id,
    )
    if active_performance is None:
        alerts.append(
            ModelAlert(
                alert_type="missing_labels",
                severity="warning",
                backend=current_backend,
                artifact_id=current_artifact_id,
                message="No recent evaluation labels are available for the active model.",
                value=0.0,
                threshold=float(selected_policy.min_evaluations),
            )
        )
    else:
        alerts.extend(
            _performance_alerts(
                active_performance,
                selected_policy,
            )
        )
    if latency_p95_ms > selected_policy.max_latency_p95_ms:
        alerts.append(
            ModelAlert(
                alert_type="latency_p95",
                severity="critical",
                backend=current_backend,
                artifact_id=current_artifact_id,
                message="p95 inference latency exceeds the serving budget.",
                value=latency_p95_ms,
                threshold=selected_policy.max_latency_p95_ms,
            )
        )
    for (backend, artifact_id, feature), count in sorted(drift_counts.items()):
        if count >= selected_policy.repeated_drift_threshold:
            alerts.append(
                ModelAlert(
                    alert_type=f"drift_{feature}",
                    severity="warning",
                    backend=backend,
                    artifact_id=artifact_id,
                    message=f"Repeated drift detected for feature '{feature}'.",
                    value=float(count),
                    threshold=float(selected_policy.repeated_drift_threshold),
                )
            )
    return alerts


def health_score(alerts: list[ModelAlert]) -> float:
    score = 100.0
    for alert in alerts:
        score -= 35.0 if alert.severity == "critical" else 15.0
    return max(0.0, score)


def alerts_to_dicts(alerts: list[ModelAlert]) -> list[dict]:
    return [
        {
            "type": alert.alert_type,
            "severity": alert.severity,
            "backend": alert.backend,
            "artifact_id": alert.artifact_id,
            "message": alert.message,
            "value": alert.value,
            "threshold": alert.threshold,
        }
        for alert in alerts
    ]


def _active_performance(
    performance: list[dict],
    backend: str,
    artifact_id: str,
) -> dict | None:
    for item in performance:
        if item.get("backend") == backend and item.get("artifact_id") == artifact_id:
            return item
    return None


def _performance_alerts(performance: dict, policy: AlertPolicy) -> list[ModelAlert]:
    backend = str(performance["backend"])
    artifact_id = str(performance["artifact_id"])
    alerts: list[ModelAlert] = []
    evaluations = int(performance.get("evaluations", 0))
    if evaluations < policy.min_evaluations:
        alerts.append(
            ModelAlert(
                alert_type="insufficient_labels",
                severity="warning",
                backend=backend,
                artifact_id=artifact_id,
                message="Evaluation label count is below the monitoring policy minimum.",
                value=float(evaluations),
                threshold=float(policy.min_evaluations),
            )
        )
    mae = float(performance.get("mae_lap_delta_s", 0.0))
    if mae > policy.max_mae_lap_delta_s:
        alerts.append(
            ModelAlert(
                alert_type="mae_lap_delta",
                severity="critical",
                backend=backend,
                artifact_id=artifact_id,
                message="Rolling lap-delta MAE exceeds the model performance budget.",
                value=mae,
                threshold=policy.max_mae_lap_delta_s,
            )
        )
    coverage = float(performance.get("interval_coverage_pct", 100.0))
    if coverage < policy.min_interval_coverage_pct:
        alerts.append(
            ModelAlert(
                alert_type="interval_coverage",
                severity="warning",
                backend=backend,
                artifact_id=artifact_id,
                message="Prediction interval coverage is below the calibration floor.",
                value=coverage,
                threshold=policy.min_interval_coverage_pct,
            )
        )
    return alerts
