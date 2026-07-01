from __future__ import annotations

import json
from pathlib import Path
from time import time
from typing import Protocol

from f1_strategy.domain import (
    OnlineFeatures,
    Prediction,
    StrategyRecommendation,
    TelemetryEvent,
)
from f1_strategy.models import FEATURE_SCHEMA_VERSION, feature_schema_hash
from f1_strategy.serialization import to_jsonable


class PersistenceStore(Protocol):
    backend_name: str

    def record_telemetry(self, event: TelemetryEvent) -> None: ...

    def record_features(self, features: OnlineFeatures) -> None: ...

    def record_prediction(self, prediction: Prediction, latency_ms: float) -> None: ...

    def record_strategy(self, recommendation: StrategyRecommendation) -> None: ...

    def record_evaluation(
        self,
        session_id: str,
        car_id: str,
        actual_lap_delta_s: float,
        prediction: Prediction,
        actual_cliff: bool | None = None,
        actual_ending_soc: float | None = None,
    ) -> None: ...

    def run_summaries(self, limit: int = 12) -> list[dict]: ...


class NullPersistence:
    backend_name = "none"

    def record_telemetry(self, event: TelemetryEvent) -> None:
        return None

    def record_features(self, features: OnlineFeatures) -> None:
        return None

    def record_prediction(self, prediction: Prediction, latency_ms: float) -> None:
        return None

    def record_strategy(self, recommendation: StrategyRecommendation) -> None:
        return None

    def record_evaluation(
        self,
        session_id: str,
        car_id: str,
        actual_lap_delta_s: float,
        prediction: Prediction,
        actual_cliff: bool | None = None,
        actual_ending_soc: float | None = None,
    ) -> None:
        return None

    def run_summaries(self, limit: int = 12) -> list[dict]:
        return []


class InMemoryPersistence:
    backend_name = "memory"

    def __init__(self) -> None:
        self.predictions: list[tuple[str, str, int, int, float, str]] = []
        self.strategies: list[tuple[str, str, int, str]] = []

    def record_telemetry(self, event: TelemetryEvent) -> None:
        return None

    def record_features(self, features: OnlineFeatures) -> None:
        return None

    def record_prediction(self, prediction: Prediction, latency_ms: float) -> None:
        self.predictions.append(
            (
                prediction.session_id,
                prediction.car_id,
                prediction.lap,
                self._now_ms(),
                latency_ms,
                self._json(prediction),
            )
        )
        self.predictions = self.predictions[-2000:]

    def record_strategy(self, recommendation: StrategyRecommendation) -> None:
        self.strategies.append(
            (
                recommendation.session_id,
                recommendation.car_id,
                self._now_ms(),
                self._json(recommendation),
            )
        )
        self.strategies = self.strategies[-500:]

    def record_evaluation(
        self,
        session_id: str,
        car_id: str,
        actual_lap_delta_s: float,
        prediction: Prediction,
        actual_cliff: bool | None = None,
        actual_ending_soc: float | None = None,
    ) -> None:
        return None

    def run_summaries(self, limit: int = 12) -> list[dict]:
        return _build_run_summaries(self.predictions, self.strategies, limit=limit)

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(to_jsonable(value), sort_keys=True)

    @staticmethod
    def _now_ms() -> int:
        return int(time() * 1000)


class DuckDBPersistence:
    backend_name = "duckdb"

    def __init__(self, path: str | Path) -> None:
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError(
                "DuckDB persistence requested but duckdb is not installed. "
                'Install with: pip install -e ".[persistence]"'
            ) from exc

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(self.path))
        self._ensure_schema()

    def record_telemetry(self, event: TelemetryEvent) -> None:
        self.connection.execute(
            """
            INSERT INTO telemetry_events
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                event.session_id,
                event.car_id,
                event.lap,
                event.sector,
                event.timestamp_ms,
                self._now_ms(),
                self._json(event),
            ],
        )

    def record_features(self, features: OnlineFeatures) -> None:
        self.connection.execute(
            """
            INSERT INTO online_features
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                features.session_id,
                features.car_id,
                features.lap,
                self._now_ms(),
                FEATURE_SCHEMA_VERSION,
                feature_schema_hash(),
                self._json(features),
            ],
        )

    def record_prediction(self, prediction: Prediction, latency_ms: float) -> None:
        self.connection.execute(
            """
            INSERT INTO predictions (
                session_id,
                car_id,
                lap,
                recorded_at_ms,
                latency_ms,
                feature_schema_version,
                feature_schema_hash,
                model_backend,
                model_artifact_id,
                app_version,
                build_sha,
                payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                prediction.session_id,
                prediction.car_id,
                prediction.lap,
                self._now_ms(),
                latency_ms,
                FEATURE_SCHEMA_VERSION,
                feature_schema_hash(),
                prediction.model_backend,
                prediction.model_artifact_id,
                prediction.app_version,
                prediction.build_sha,
                self._json(prediction),
            ],
        )

    def record_strategy(self, recommendation: StrategyRecommendation) -> None:
        self.connection.execute(
            """
            INSERT INTO strategies
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                recommendation.session_id,
                recommendation.car_id,
                recommendation.prediction.lap,
                self._now_ms(),
                self._json(recommendation),
            ],
        )

    def record_evaluation(
        self,
        session_id: str,
        car_id: str,
        actual_lap_delta_s: float,
        prediction: Prediction,
        actual_cliff: bool | None = None,
        actual_ending_soc: float | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO evaluations (
                session_id,
                car_id,
                lap,
                recorded_at_ms,
                actual_lap_delta_s,
                actual_cliff,
                actual_ending_soc,
                model_backend,
                model_artifact_id,
                app_version,
                build_sha,
                prediction_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                session_id,
                car_id,
                prediction.lap,
                self._now_ms(),
                actual_lap_delta_s,
                actual_cliff,
                actual_ending_soc,
                prediction.model_backend,
                prediction.model_artifact_id,
                prediction.app_version,
                prediction.build_sha,
                self._json(prediction),
            ],
        )

    def run_summaries(self, limit: int = 12) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT session_id, car_id, lap, recorded_at_ms, latency_ms, payload_json
            FROM predictions
            WHERE session_id LIKE 'sim-race-%'
            ORDER BY recorded_at_ms DESC
            LIMIT 1000
            """
        ).fetchall()
        strategies = self.connection.execute(
            """
            SELECT session_id, car_id, recorded_at_ms, payload_json
            FROM strategies
            WHERE session_id LIKE 'sim-race-%'
            ORDER BY recorded_at_ms DESC
            LIMIT 1000
            """
        ).fetchall()

        return _build_run_summaries(rows, strategies, limit=limit)

    def close(self) -> None:
        self.connection.close()

    def _ensure_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry_events (
                session_id TEXT,
                car_id TEXT,
                lap INTEGER,
                sector INTEGER,
                event_timestamp_ms BIGINT,
                recorded_at_ms BIGINT,
                payload_json TEXT
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS online_features (
                session_id TEXT,
                car_id TEXT,
                lap INTEGER,
                recorded_at_ms BIGINT,
                feature_schema_version TEXT,
                feature_schema_hash TEXT,
                payload_json TEXT
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                session_id TEXT,
                car_id TEXT,
                lap INTEGER,
                recorded_at_ms BIGINT,
                latency_ms DOUBLE,
                feature_schema_version TEXT,
                feature_schema_hash TEXT,
                model_backend TEXT,
                model_artifact_id TEXT,
                app_version TEXT,
                build_sha TEXT,
                payload_json TEXT
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS strategies (
                session_id TEXT,
                car_id TEXT,
                lap INTEGER,
                recorded_at_ms BIGINT,
                payload_json TEXT
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                session_id TEXT,
                car_id TEXT,
                lap INTEGER,
                recorded_at_ms BIGINT,
                actual_lap_delta_s DOUBLE,
                actual_cliff BOOLEAN,
                actual_ending_soc DOUBLE,
                model_backend TEXT,
                model_artifact_id TEXT,
                app_version TEXT,
                build_sha TEXT,
                prediction_json TEXT
            )
            """
        )
        self._ensure_column("predictions", "model_backend", "TEXT")
        self._ensure_column("predictions", "model_artifact_id", "TEXT")
        self._ensure_column("predictions", "app_version", "TEXT")
        self._ensure_column("predictions", "build_sha", "TEXT")
        self._ensure_column("evaluations", "model_backend", "TEXT")
        self._ensure_column("evaluations", "model_artifact_id", "TEXT")
        self._ensure_column("evaluations", "app_version", "TEXT")
        self._ensure_column("evaluations", "build_sha", "TEXT")

    def export_parquet(self, output_dir: Path | str) -> dict[str, Path]:
        """Export all tables to Parquet files in output_dir. Returns {table: path}."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tables = ["telemetry_events", "online_features", "predictions", "strategies", "evaluations"]
        exported: dict[str, Path] = {}
        for table in tables:
            out = output_dir / f"{table}.parquet"
            self.connection.execute(
                f"COPY (SELECT * FROM {table}) TO '{out}' (FORMAT PARQUET)"
            )
            exported[table] = out
        return exported

    def _ensure_column(self, table: str, column: str, data_type: str) -> None:
        self.connection.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {data_type}"
        )

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(to_jsonable(value), sort_keys=True)

    @staticmethod
    def _now_ms() -> int:
        return int(time() * 1000)


def create_persistence_store(backend: str, duckdb_path: str) -> PersistenceStore:
    normalized = backend.strip().lower()
    if normalized in {"", "none", "off", "disabled"}:
        return NullPersistence()
    if normalized == "auto":
        try:
            return DuckDBPersistence(duckdb_path)
        except Exception:
            return InMemoryPersistence()
    if normalized in {"memory", "in-memory", "inmemory"}:
        return InMemoryPersistence()
    if normalized == "duckdb":
        return DuckDBPersistence(duckdb_path)
    raise ValueError(f"Unsupported persistence backend: {backend}")


def _build_run_summaries(
    prediction_rows: list[tuple],
    strategy_rows: list[tuple],
    limit: int = 12,
) -> list[dict]:
    latest_strategy_by_run: dict[tuple[str, str], dict] = {}
    for session_id, car_id, recorded_at_ms, payload_json in strategy_rows:
        key = (session_id, car_id)
        if key in latest_strategy_by_run:
            continue
        payload = json.loads(payload_json)
        latest_strategy_by_run[key] = {
            "recorded_at_ms": recorded_at_ms,
            "pit_target_lap": payload["pit_window"]["target_lap"],
            "pit_window": {
                "earliest_lap": payload["pit_window"]["earliest_lap"],
                "latest_lap": payload["pit_window"]["latest_lap"],
            },
            "pace_target_delta_s": payload["pace_target_delta_s"],
        }

    summaries: dict[tuple[str, str], dict] = {}
    for session_id, car_id, lap, recorded_at_ms, latency_ms, payload_json in prediction_rows:
        if not str(session_id).startswith("sim-race-"):
            continue
        key = (session_id, car_id)
        payload = json.loads(payload_json)
        payload["_recorded_at_ms"] = recorded_at_ms
        summary = summaries.setdefault(
            key,
            {
                "session_id": session_id,
                "car_id": car_id,
                "started_at_ms": recorded_at_ms,
                "updated_at_ms": recorded_at_ms,
                "prediction_count": 0,
                "latest_lap": 0,
                "max_tire_wear_pct": 0.0,
                "max_cliff_probability": 0.0,
                "max_overheating_probability": 0.0,
                "avg_lap_delta_s": 0.0,
                "min_lap_delta_s": float("inf"),
                "max_lap_delta_s": float("-inf"),
                "avg_ers_efficiency": 0.0,
                "avg_prediction_interval_width_s": 0.0,
                "avg_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "latest_prediction": payload.copy(),
                "latest_strategy": latest_strategy_by_run.get(key),
            },
        )
        summary["started_at_ms"] = min(summary["started_at_ms"], recorded_at_ms)
        summary["updated_at_ms"] = max(summary["updated_at_ms"], recorded_at_ms)
        summary["prediction_count"] += 1
        summary["latest_lap"] = max(summary["latest_lap"], lap)
        summary["max_tire_wear_pct"] = max(summary["max_tire_wear_pct"], float(payload["tire_wear_pct"]))
        summary["max_cliff_probability"] = max(
            summary["max_cliff_probability"],
            float(payload["cliff_probability"]),
        )
        summary["max_overheating_probability"] = max(
            summary["max_overheating_probability"],
            float(payload["overheating_probability"]),
        )
        lap_delta = float(payload["next_lap_delta_s"])
        interval_width = float(payload["uncertainty_high_s"]) - float(payload["uncertainty_low_s"])
        summary["avg_lap_delta_s"] += lap_delta
        summary["min_lap_delta_s"] = min(summary["min_lap_delta_s"], lap_delta)
        summary["max_lap_delta_s"] = max(summary["max_lap_delta_s"], lap_delta)
        summary["avg_ers_efficiency"] += float(payload["ers_efficiency"])
        summary["avg_prediction_interval_width_s"] += interval_width
        summary["avg_latency_ms"] += float(latency_ms)
        summary["max_latency_ms"] = max(summary["max_latency_ms"], float(latency_ms))
        if recorded_at_ms >= summary["latest_prediction"].get("_recorded_at_ms", 0):
            summary["latest_prediction"] = payload

    ordered = sorted(summaries.values(), key=lambda item: item["updated_at_ms"], reverse=True)
    for summary in ordered:
        count = max(1, summary["prediction_count"])
        summary["avg_lap_delta_s"] = summary["avg_lap_delta_s"] / count
        summary["avg_latency_ms"] = summary["avg_latency_ms"] / count
        summary["avg_ers_efficiency"] = summary["avg_ers_efficiency"] / count
        summary["avg_prediction_interval_width_s"] = (
            summary["avg_prediction_interval_width_s"] / count
        )
        if summary["min_lap_delta_s"] == float("inf"):
            summary["min_lap_delta_s"] = 0.0
        if summary["max_lap_delta_s"] == float("-inf"):
            summary["max_lap_delta_s"] = 0.0
        summary["latest_prediction"].pop("_recorded_at_ms", None)
    return ordered[: max(1, limit)]



def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Export DuckDB tables to Parquet files.")
    parser.add_argument("--db", default="data/f1_strategy.duckdb", help="Path to DuckDB database")
    parser.add_argument("--output", default="data/exports", help="Output directory for Parquet files")
    args = parser.parse_args()
    store = DuckDBPersistence(args.db)
    exported = store.export_parquet(args.output)
    for table, path in exported.items():
        print(f"  {table}: {path}")
