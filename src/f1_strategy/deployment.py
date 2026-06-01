from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from f1_strategy.models import feature_schema_hash


@dataclass(frozen=True)
class DeploymentReadiness:
    ready: bool
    active_backend: str
    active_artifact_id: str
    promoted: bool
    health_score: float
    alert_count: int
    critical_alert_count: int
    feature_schema_match: bool
    persistence_backend: str
    persistence_ready: bool
    latency_p95_ms: float
    target_latency_ms: float
    rollback_candidate: dict[str, Any] | None
    checks: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "active_backend": self.active_backend,
            "active_artifact_id": self.active_artifact_id,
            "promoted": self.promoted,
            "health_score": self.health_score,
            "alert_count": self.alert_count,
            "critical_alert_count": self.critical_alert_count,
            "feature_schema_match": self.feature_schema_match,
            "persistence_backend": self.persistence_backend,
            "persistence_ready": self.persistence_ready,
            "latency_p95_ms": self.latency_p95_ms,
            "target_latency_ms": self.target_latency_ms,
            "rollback_candidate": self.rollback_candidate,
            "checks": self.checks,
        }


def deployment_readiness(
    active_backend: str,
    active_artifact_id: str,
    artifact_root: str | Path,
    alerts: dict[str, Any],
    latency_p95_ms: float,
    target_latency_ms: float,
    persistence_backend: str,
) -> DeploymentReadiness:
    registry = load_registry(artifact_root)
    promoted = is_promoted(registry, active_backend, active_artifact_id)
    schema_match = active_artifact_id == "unregistered" or artifact_schema_matches(
        artifact_root,
        active_artifact_id,
    )
    alert_items = list(alerts.get("alerts", []))
    critical_count = sum(1 for alert in alert_items if alert.get("severity") == "critical")
    persistence_ready = persistence_backend not in {"unknown", "none"}
    latency_ready = latency_p95_ms <= target_latency_ms
    health_score = float(alerts.get("health_score", 0.0))
    checks = {
        "promoted_or_local": promoted or active_artifact_id == "unregistered",
        "feature_schema_match": schema_match,
        "no_critical_alerts": critical_count == 0,
        "health_score": health_score >= 70.0,
        "persistence_ready": persistence_ready,
        "latency_ready": latency_ready,
    }
    ready = all(checks.values())
    return DeploymentReadiness(
        ready=ready,
        active_backend=active_backend,
        active_artifact_id=active_artifact_id,
        promoted=promoted,
        health_score=health_score,
        alert_count=len(alert_items),
        critical_alert_count=critical_count,
        feature_schema_match=schema_match,
        persistence_backend=persistence_backend,
        persistence_ready=persistence_ready,
        latency_p95_ms=latency_p95_ms,
        target_latency_ms=target_latency_ms,
        rollback_candidate=rollback_candidate(
            registry,
            backend=active_backend,
            active_artifact_id=active_artifact_id,
        ),
        checks=checks,
    )


def rollback_candidate(
    registry: dict[str, Any],
    backend: str,
    active_artifact_id: str,
) -> dict[str, Any] | None:
    artifacts = [
        artifact
        for artifact in registry.get("artifacts", [])
        if artifact.get("backend") == backend
        and artifact.get("status") == "promoted"
        and artifact.get("artifact_id") != active_artifact_id
        and artifact.get("feature_schema_hash") == feature_schema_hash()
    ]
    artifacts.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return artifacts[0] if artifacts else None


def latest_promoted_artifact(
    registry: dict[str, Any],
    backend: str | None = None,
) -> dict[str, Any] | None:
    promoted_ids = set(registry.get("promoted", {}).values())
    artifacts = [
        artifact
        for artifact in registry.get("artifacts", [])
        if artifact.get("status") == "promoted"
        and artifact.get("artifact_id") in promoted_ids
        and artifact.get("feature_schema_hash") == feature_schema_hash()
        and (backend is None or artifact.get("backend") == backend)
    ]
    artifacts.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return artifacts[0] if artifacts else None


def load_registry(artifact_root: str | Path) -> dict[str, Any]:
    path = Path(artifact_root) / "registry.json"
    if not path.exists():
        return {"artifacts": [], "promoted": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def is_promoted(registry: dict[str, Any], backend: str, artifact_id: str) -> bool:
    if artifact_id == "unregistered":
        return False
    if registry.get("promoted", {}).get(backend) == artifact_id:
        return True
    return any(
        artifact.get("artifact_id") == artifact_id and artifact.get("status") == "promoted"
        for artifact in registry.get("artifacts", [])
    )


def artifact_schema_matches(artifact_root: str | Path, artifact_id: str) -> bool:
    manifest_path = Path(artifact_root) / artifact_id / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("feature_schema_hash") == feature_schema_hash()
