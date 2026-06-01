from __future__ import annotations

import os
from dataclasses import dataclass


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


@dataclass(frozen=True)
class Settings:
    env: str = os.getenv("F1_ENV", "local")
    log_level: str = os.getenv("F1_LOG_LEVEL", "INFO")
    api_host: str = os.getenv("F1_API_HOST", "0.0.0.0")
    api_port: int = _get_int("F1_API_PORT", 8000)
    feature_window_size: int = _get_int("F1_FEATURE_WINDOW_SIZE", 50)
    model_backend: str = os.getenv("F1_MODEL_BACKEND", "auto")
    model_artifact_id: str = os.getenv("F1_MODEL_ARTIFACT_ID", "")
    model_artifact_root: str = os.getenv("F1_MODEL_ARTIFACT_ROOT", "artifacts/models")
    xgboost_model_path: str = os.getenv("F1_XGBOOST_MODEL_PATH", "models/xgboost_lap_delta.json")
    lightgbm_model_path: str = os.getenv("F1_LIGHTGBM_MODEL_PATH", "models/lightgbm_lap_delta.txt")
    catboost_model_path: str = os.getenv("F1_CATBOOST_MODEL_PATH", "models/catboost_lap_delta.cbm")
    sequence_model_path: str = os.getenv("F1_SEQUENCE_MODEL_PATH", "models/sequence_lap_delta.pt")
    target_latency_ms: float = _get_float("F1_TARGET_LATENCY_MS", 25.0)
    base_lap_time_s: float = _get_float("F1_BASE_LAP_TIME_S", 90.0)
    pit_loss_s: float = _get_float("F1_PIT_LOSS_S", 21.0)
    drift_threshold_z: float = _get_float("F1_DRIFT_THRESHOLD_Z", 3.0)
    max_temporal_oscillation_s: float = _get_float("F1_MAX_TEMPORAL_OSCILLATION_S", 1.25)
    max_calibration_width_s: float = _get_float("F1_MAX_CALIBRATION_WIDTH_S", 1.80)
    kafka_bootstrap_servers: str = os.getenv("F1_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    redis_url: str = os.getenv("F1_REDIS_URL", "redis://localhost:6379/0")
    clickhouse_dsn: str = os.getenv("F1_CLICKHOUSE_DSN", "http://localhost:8123/default")
    mlflow_tracking_uri: str = os.getenv("F1_MLFLOW_TRACKING_URI", "http://localhost:5000")
    prometheus_enabled: bool = _get_bool("F1_PROMETHEUS_ENABLED", True)
    persistence_backend: str = os.getenv("F1_PERSISTENCE_BACKEND", "auto")
    duckdb_path: str = os.getenv("F1_DUCKDB_PATH", "data/f1_strategy.duckdb")


def load_settings() -> Settings:
    return Settings()
