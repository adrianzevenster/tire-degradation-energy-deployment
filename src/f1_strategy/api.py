from __future__ import annotations

import csv
import json
import threading
import urllib.error
import urllib.request
import uuid as _uuid
from dataclasses import replace
from pathlib import Path
from time import time as _time

from f1_strategy.artifacts import artifact_release_detail
from f1_strategy.config import load_settings
from f1_strategy.data_sources.fastf1_export import (
    FastF1ExportConfig,
    export_fastf1_replay,
    manifest_path_for,
)
from f1_strategy.data_sources.openf1_export import (
    OpenF1ExportConfig,
    export_openf1_session,
)
from f1_strategy.deployment import load_registry
from f1_strategy.engine import InferenceEngine
from f1_strategy.live import LiveSimulationManager
from f1_strategy.metadata import APP_VERSION, build_info
from f1_strategy.models import (
    FEATURE_SCHEMA_VERSION,
    ModelConfig,
    create_serving_model,
    feature_schema_hash,
)
from f1_strategy.monitoring import monitoring_catalog
from f1_strategy.replay import (
    DEFAULT_REPLAY_DATASET,
    run_benchmark_replay_suite,
    replay_data_provenance,
    replay_report_to_dict,
    replay_suite_to_dict,
    run_replay_evaluation,
    run_replay_suite,
)
from f1_strategy.regression import RegressionConfig, RegressionSuite
from f1_strategy.data_sources.live_timing import LiveStreamManager
from f1_strategy.serialization import telemetry_from_dict, to_jsonable

engine = InferenceEngine()
settings = load_settings()
simulation = LiveSimulationManager()
live_stream = LiveStreamManager()
runtime_build = build_info()


def _live_ingest(event: object) -> None:
    """Thread-safe callback from the live stream into the engine."""
    try:
        engine.ingest(event)  # type: ignore[arg-type]
    except Exception:
        pass


live_stream.set_event_callback(_live_ingest)
MODEL_BACKENDS = [
    "auto",
    "hybrid",
    "kalman",
    "xgboost",
    "lightgbm",
    "catboost",
    "sequence",
    "river",
]

try:
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field, field_validator

    app = FastAPI(title="F1 Tire and Energy Strategy API", version=APP_VERSION)
    ui_dir = Path(__file__).with_name("ui")
    app.mount("/ui", StaticFiles(directory=ui_dir), name="ui")

    _training_jobs: dict[str, dict] = {}

    def _shadow_auto_promote_worker() -> None:
        global engine
        import time
        while True:
            time.sleep(30)
            try:
                candidate = engine.shadow.promotion_candidate()
                if candidate and candidate.get("recommendation") == "promote_challenger":
                    artifact_id = candidate.get("challenger_artifact", "unregistered")
                    if artifact_id and artifact_id != "unregistered":
                        old = engine
                        engine = InferenceEngine(
                            settings=replace(settings, model_backend="auto", model_artifact_id=artifact_id),
                            monitoring=old.monitoring,
                        )
                        old.shadow.disable()
            except Exception:
                pass

    threading.Thread(target=_shadow_auto_promote_worker, daemon=True).start()

    class TelemetryEventRequest(BaseModel):
        session_id: str = Field(..., min_length=1, examples=["sim-race"])
        car_id: str = Field(..., min_length=1, examples=["car-16"])
        lap: int = Field(..., ge=1, examples=[12])
        sector: int = Field(..., ge=1, le=3, examples=[2])
        speed_kph: float = Field(..., ge=0.0, examples=[241.5])
        throttle: float = Field(..., ge=0.0, le=1.0, examples=[0.82])
        brake: float = Field(..., ge=0.0, le=1.0, examples=[0.34])
        steering_angle: float = Field(..., examples=[-8.5])
        tire_temp_fl: float = Field(..., examples=[96.2])
        tire_temp_fr: float = Field(..., examples=[95.8])
        tire_temp_rl: float = Field(..., examples=[92.6])
        tire_temp_rr: float = Field(..., examples=[93.1])
        brake_temp: float = Field(..., ge=0.0, examples=[710.0])
        slip_angle: float = Field(..., examples=[3.4])
        lateral_g: float = Field(..., ge=0.0, examples=[3.1])
        ers_soc: float = Field(..., ge=0.0, le=1.0, examples=[0.64])
        ers_deployment_kw: float = Field(..., ge=0.0, examples=[78.0])
        fuel_kg: float = Field(..., ge=0.0, examples=[48.5])
        track_temp_c: float = Field(..., examples=[39.2])
        air_temp_c: float = Field(..., examples=[27.4])
        humidity: float = Field(..., ge=0.0, le=1.0, examples=[0.44])
        compound: str = Field(..., examples=["medium"])
        lap_time_s: float | None = Field(default=None, ge=0.0, examples=[90.42])
        timestamp_ms: int | None = Field(default=None, ge=0, examples=[123000])

    class EvaluationRequest(BaseModel):
        session_id: str = Field(..., min_length=1, examples=["sim-race"])
        car_id: str = Field(..., min_length=1, examples=["car-16"])
        actual_lap_delta_s: float = Field(..., examples=[0.27])
        actual_cliff: bool | None = None
        actual_ending_soc: float | None = Field(default=None, ge=0.0, le=1.0)

    class FastF1ExportRequest(BaseModel):
        year: int = Field(..., ge=2018, le=2100, examples=[2024])
        event: str = Field(..., min_length=1, examples=["Bahrain"])
        session: str = Field(..., min_length=1, examples=["R"])
        driver: str = Field(..., min_length=1, examples=["VER"])
        output: str | None = Field(default=None, examples=["data/fastf1-2024-bahrain-r-ver.csv"])
        cache_dir: str | None = Field(default="data/fastf1-cache")
        max_laps: int | None = Field(default=None, ge=1, le=100)

    class OpenF1ExportRequest(BaseModel):
        year: int = Field(..., ge=2018, le=2100, examples=[2024])
        event: str = Field(..., min_length=1, examples=["Bahrain"])
        session: str = Field(default="Race", min_length=1, examples=["Race"])
        driver: str = Field(..., min_length=1, examples=["VER"])

    class StatusResponse(BaseModel):
        status: str

    class TrainingRequest(BaseModel):
        backend: str = Field(default="xgboost", examples=["xgboost"])
        laps: int = Field(default=28, ge=1, le=200)
        seeds: int = Field(default=64, ge=1, le=500)
        rounds: int = Field(default=140, ge=10, le=1000)
        real_data: list[str] | None = Field(
            default=None,
            examples=[["data/fastf1-2024-bahrain-r-ver.csv", "data/fastf1-2024-monaco-r-ver.csv"]],
        )
        use_mlflow: bool = Field(default=False)
        register_artifact: bool = Field(default=True)
        base_lap_time_s: float | None = Field(default=None, ge=60.0, le=200.0, examples=[96.0])

        @field_validator("real_data", mode="before")
        @classmethod
        def _coerce_real_data(cls, v: object) -> object:
            if isinstance(v, str):
                return [v]
            return v

    def _payload(model: BaseModel) -> dict:
        if hasattr(model, "model_dump"):
            return model.model_dump(exclude_none=True)
        return model.dict(exclude_none=True)

    def _probe_service(url: str, timeout_s: float = 0.35) -> str:
        if not url:
            return "not_configured"
        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                if 200 <= response.status < 500:
                    return "available"
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return "unavailable"
        return "unavailable"

    def _external_service_links(probe: bool = False) -> list[dict[str, str | bool]]:
        external = [
            {
                "id": "mlflow",
                "label": "MLflow",
                "url": settings.mlflow_ui_url,
                "hint": "Start the MLflow sidecar or set F1_MLFLOW_UI_URL.",
            },
            {
                "id": "grafana",
                "label": "Grafana",
                "url": settings.grafana_url,
                "hint": "Start the Grafana sidecar or set F1_GRAFANA_URL.",
            },
            {
                "id": "prometheus",
                "label": "Prometheus",
                "url": settings.prometheus_url,
                "hint": "Start the Prometheus sidecar or set F1_PROMETHEUS_URL.",
            },
        ]
        services: list[dict[str, str | bool]] = []
        for service in external:
            status = _probe_service(str(service["url"])) if probe else "configured"
            services.append({**service, "status": status, "external": True})
        services.extend(
            [
                {
                    "id": "api-docs",
                    "label": "API Docs",
                    "url": "/docs",
                    "status": "available",
                    "external": False,
                    "hint": "FastAPI documentation served by this app.",
                },
                {
                    "id": "metrics",
                    "label": "Metrics",
                    "url": "/metrics",
                    "status": "available" if settings.prometheus_enabled else "disabled",
                    "external": False,
                    "hint": "Prometheus-format metrics served by this app.",
                },
            ]
        )
        return services

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(ui_dir / "index.html")

    @app.get("/integrations/external-links")
    def external_links(probe: bool = False) -> dict[str, list[dict[str, str | bool]]]:
        return {"services": _external_service_links(probe=probe)}

    @app.get("/health")
    def health() -> dict[str, float | int | str | bool]:
        drift_fitted = len(engine.drift_detector._baseline) > 0
        feature_store_backend = getattr(engine.feature_store, "backend_name", "memory")
        return {
            "status": "ok",
            "version": runtime_build.version,
            "build_sha": runtime_build.build_sha,
            "build_date": runtime_build.build_date,
            "env": settings.env,
            "model_backend": getattr(engine.model, "backend_name", engine.model.__class__.__name__),
            "model_artifact_id": engine.settings.model_artifact_id or "unregistered",
            "persistence_backend": getattr(engine.persistence, "backend_name", "unknown"),
            "feature_store_backend": feature_store_backend,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema_hash": feature_schema_hash(),
            "latency_p95_ms": round(engine.latency_p95_ms(), 4),
            "target_latency_ms": settings.target_latency_ms,
            "feature_window_size": settings.feature_window_size,
            "drift_baseline_fitted": drift_fitted,
            "drift_ingest_count": engine._ingest_count,
        }

    @app.get("/version")
    def version() -> dict[str, str]:
        return {
            "version": runtime_build.version,
            "build_sha": runtime_build.build_sha,
            "build_date": runtime_build.build_date,
        }

    @app.get("/models")
    def models() -> dict[str, list[str] | str]:
        return {
            "active_backend": getattr(
                engine.model,
                "backend_name",
                engine.model.__class__.__name__,
            ),
            "configured_backend": engine.settings.model_backend,
            "active_artifact_id": engine.settings.model_artifact_id or "unregistered",
            "available_backends": MODEL_BACKENDS,
        }

    @app.get("/artifacts")
    def artifacts() -> dict:
        registry = load_registry(engine.settings.model_artifact_root)
        return {
            "artifact_root": engine.settings.model_artifact_root,
            "active_artifact_id": engine.settings.model_artifact_id or "unregistered",
            "promoted": registry.get("promoted", {}),
            "artifacts": registry.get("artifacts", []),
        }

    @app.get("/artifacts/{artifact_id:path}")
    def artifact_detail(artifact_id: str) -> dict:
        try:
            return artifact_release_detail(
                artifact_id=artifact_id,
                artifact_root=engine.settings.model_artifact_root,
            )
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/model/backend")
    def model_backend(backend: str) -> dict[str, str]:
        global engine
        normalized = backend.strip().lower()
        if normalized not in MODEL_BACKENDS:
            raise HTTPException(status_code=400, detail=f"Unsupported model backend: {backend}")
        old_engine = engine
        try:
            engine = InferenceEngine(
                settings=replace(
                    settings,
                    model_backend=normalized,
                    model_artifact_id="",
                ),
                monitoring=old_engine.monitoring,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if hasattr(old_engine.persistence, "close"):
            old_engine.persistence.close()
        simulation.reset()
        return {
            "configured_backend": normalized,
            "active_backend": getattr(
                engine.model,
                "backend_name",
                engine.model.__class__.__name__,
            ),
            "active_artifact_id": "unregistered",
        }

    @app.post("/model/artifact")
    def model_artifact(artifact_id: str) -> dict[str, str]:
        global engine
        normalized = artifact_id.strip()
        if not normalized:
            raise HTTPException(status_code=400, detail="artifact_id is required")
        old_engine = engine
        try:
            engine = InferenceEngine(
                settings=replace(
                    settings,
                    model_backend="auto",
                    model_artifact_id=normalized,
                ),
                monitoring=old_engine.monitoring,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if hasattr(old_engine.persistence, "close"):
            old_engine.persistence.close()
        simulation.reset()
        return {
            "configured_backend": engine.settings.model_backend,
            "active_backend": getattr(
                engine.model,
                "backend_name",
                engine.model.__class__.__name__,
            ),
            "active_artifact_id": engine.settings.model_artifact_id,
        }

    @app.post("/shadow/configure")
    def shadow_configure(backend: str, artifact_id: str = "") -> dict:
        normalized = backend.strip().lower()
        if normalized not in MODEL_BACKENDS:
            raise HTTPException(status_code=400, detail=f"Unsupported backend: {backend}")
        config = ModelConfig(
            target_latency_ms=settings.target_latency_ms,
            base_lap_time_s=settings.base_lap_time_s,
            pit_loss_s=settings.pit_loss_s,
        )
        challenger_settings = replace(
            settings,
            model_backend=normalized,
            model_artifact_id=artifact_id.strip(),
        )
        try:
            challenger = create_serving_model(
                config=config,
                backend=challenger_settings.model_backend,
                xgboost_model_path=challenger_settings.xgboost_model_path,
                lightgbm_model_path=challenger_settings.lightgbm_model_path,
                catboost_model_path=challenger_settings.catboost_model_path,
                sequence_model_path=challenger_settings.sequence_model_path,
                model_artifact_id=challenger_settings.model_artifact_id,
                model_artifact_root=challenger_settings.model_artifact_root,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        engine.shadow.configure(
            model=challenger,
            backend=normalized,
            artifact_id=artifact_id.strip() or "unregistered",
        )
        return engine.shadow.status()

    @app.get("/shadow/status")
    def shadow_status() -> dict:
        return engine.shadow.status()

    @app.delete("/shadow")
    def shadow_disable() -> dict[str, str]:
        engine.shadow.disable()
        return {"status": "shadow disabled"}

    @app.get("/shadow/promotion-candidate")
    def shadow_promotion_candidate() -> dict:
        candidate = engine.shadow.promotion_candidate()
        return {"candidate": candidate, "active": engine.shadow.active}

    class ReplayStartRequest(BaseModel):
        dataset_path: str = Field(..., min_length=1, examples=["data/fastf1-2024-bahrain-r-ver.csv"])
        speed_multiplier: float = Field(default=1.0, ge=0.1, le=100.0, examples=[1.0])

    class LiveStartRequest(BaseModel):
        driver: str = Field(..., min_length=1, examples=["VER"])
        session_id: str = Field(default="live-session", min_length=1)
        recording_path: str = Field(default="data/live-timing-recording.txt")
        no_auth: bool = Field(default=False)
        timeout: int = Field(default=120, ge=10, le=7200)

    class ReplayRunRequest(BaseModel):
        kind: str = Field(default="dataset", examples=["dataset", "suite", "benchmark"])
        dataset_path: str = Field(default=str(DEFAULT_REPLAY_DATASET), min_length=1)

    class RegressionRunRequest(BaseModel):
        laps: int = Field(default=18, ge=1, le=200)
        seed: int = Field(default=11, ge=1, le=9999)
        target_latency_ms: float | None = Field(default=None, ge=0.0)
        max_temporal_oscillation_s: float | None = Field(default=None, ge=0.0)
        min_calibration_width_s: float = Field(default=0.20, ge=0.0)
        max_calibration_width_s: float | None = Field(default=None, ge=0.0)

    def _stream_status_dict() -> dict:
        s = live_stream.status()
        return {
            "mode": s.mode,
            "connected": s.connected,
            "session_id": s.session_id,
            "driver": s.driver,
            "events_ingested": s.events_ingested,
            "events_per_second": round(s.events_per_second, 2),
            "latest_lap": s.latest_lap,
            "latest_lap_time_s": s.latest_lap_time_s,
            "current_compound": s.current_compound,
            "dataset_path": s.dataset_path,
            "message_count": s.message_count,
            "speed_multiplier": s.speed_multiplier,
            "progress_pct": s.progress_pct,
            "error": s.error,
        }

    @app.post("/live-data/replay/start")
    def live_replay_start(payload: ReplayStartRequest) -> dict:
        dataset = _safe_dataset_path(payload.dataset_path)
        live_stream.configure_replay(
            dataset_path=str(dataset),
            speed_multiplier=payload.speed_multiplier,
        )
        try:
            live_stream.start()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _stream_status_dict()

    @app.post("/live-data/live/start")
    def live_timing_start(payload: LiveStartRequest) -> dict:
        live_stream.configure_live(
            driver=payload.driver,
            session_id=payload.session_id,
            recording_path=payload.recording_path,
            no_auth=payload.no_auth,
            timeout=payload.timeout,
        )
        try:
            live_stream.start()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _stream_status_dict()

    @app.get("/live-data/status")
    def live_data_status() -> dict:
        return _stream_status_dict()

    @app.delete("/live-data")
    def live_data_stop() -> dict[str, str]:
        live_stream.stop()
        return {"status": "stopped"}

    @app.get("/monitoring/catalog")
    def catalog() -> dict[str, list[str]]:
        return monitoring_catalog()

    @app.get("/monitoring/model-performance")
    def model_performance() -> dict[str, list[dict]]:
        return {"models": engine.model_performance()}

    @app.get("/monitoring/model-comparison")
    def model_comparison() -> dict[str, list[dict]]:
        return {"models": engine.model_comparison()}

    @app.get("/monitoring/alerts")
    def monitoring_alerts() -> dict:
        return engine.model_alerts()

    @app.get("/deployment/readiness")
    def deployment_readiness(mode: str = "local") -> dict:
        if mode not in {"local", "production"}:
            raise HTTPException(status_code=400, detail="mode must be local or production")
        return engine.deployment_readiness(mode=mode)

    @app.get("/deployment/rollback-candidate")
    def deployment_rollback_candidate() -> dict:
        readiness = engine.deployment_readiness()
        candidate = readiness.get("rollback_candidate")
        return {"candidate": candidate}

    @app.get("/metrics")
    def metrics() -> Response:
        engine.deployment_readiness()
        return Response(engine.monitoring.render_prometheus(), media_type="text/plain")

    @app.post("/simulation/start")
    def simulation_start(laps: int = 18, seed: int = 7) -> dict[str, int | bool | str]:
        simulation.start(laps=laps, seed=seed)
        return simulation.status()

    @app.post("/simulation/stop")
    def simulation_stop() -> dict[str, int | bool | str]:
        simulation.stop()
        return simulation.status()

    @app.post("/simulation/reset")
    def simulation_reset() -> dict[str, int | bool | str]:
        simulation.reset()
        return simulation.status()

    @app.post("/simulation/tick")
    def simulation_tick(batch_size: int = 1, remaining_laps: int = 30) -> dict:
        predictions = []
        telemetry = []
        strategy = None
        for event in simulation.tick(batch_size=batch_size):
            telemetry.append(to_jsonable(event))
            prediction = engine.ingest(event)
            predictions.append(to_jsonable(prediction))
            strategy = engine.strategy(
                event.session_id,
                event.car_id,
                remaining_laps=remaining_laps,
            )
        return {
            "status": simulation.status(),
            "telemetry": telemetry,
            "predictions": predictions,
            "strategy": to_jsonable(strategy) if strategy is not None else None,
            "metrics": engine.monitoring.render_prometheus(),
        }

    @app.get("/simulation/status")
    def simulation_status() -> dict[str, int | bool | str]:
        return simulation.status()

    @app.get("/history/runs")
    def history_runs(limit: int = 12) -> dict[str, list[dict] | str]:
        return {
            "persistence_backend": getattr(engine.persistence, "backend_name", "unknown"),
            "runs": engine.persistence.run_summaries(limit=limit),
        }

    @app.post("/telemetry")
    def ingest(payload: TelemetryEventRequest) -> dict:
        try:
            prediction = engine.ingest(telemetry_from_dict(_payload(payload)))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return to_jsonable(prediction)

    @app.post("/evaluation", response_model=StatusResponse)
    def evaluation(payload: EvaluationRequest) -> dict[str, str]:
        try:
            request = _payload(payload)
            engine.record_evaluation(
                session_id=request["session_id"],
                car_id=request["car_id"],
                actual_lap_delta_s=float(request["actual_lap_delta_s"]),
                actual_cliff=request.get("actual_cliff"),
                actual_ending_soc=request.get("actual_ending_soc"),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "recorded"}

    @app.get("/evaluation/replay")
    def replay_evaluation(dataset_path: str = str(DEFAULT_REPLAY_DATASET)) -> dict:
        try:
            report = run_replay_evaluation(_safe_dataset_path(dataset_path), engine=engine)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return replay_report_to_dict(report)

    @app.get("/data-sources/replay-datasets")
    def replay_datasets() -> dict[str, list[dict]]:
        try:
            datasets = _list_replay_datasets()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"datasets": datasets}

    @app.get("/data-sources/replay-datasets/{dataset_path:path}/manifest")
    def replay_dataset_manifest(dataset_path: str) -> dict:
        try:
            dataset = _safe_dataset_path(dataset_path)
            manifest_path = manifest_path_for(dataset)
            if not manifest_path.exists():
                raise FileNotFoundError(f"Replay dataset manifest does not exist: {manifest_path}")
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/data-sources/fastf1/export")
    def fastf1_export(payload: FastF1ExportRequest) -> dict:
        try:
            request = _payload(payload)
            output = _safe_data_output_path(
                request.get("output") or _default_fastf1_output(request)
            )
            manifest = export_fastf1_replay(
                FastF1ExportConfig(
                    year=int(request["year"]),
                    event=str(request["event"]),
                    session=str(request["session"]),
                    driver=str(request["driver"]),
                    output=output,
                    cache_dir=_safe_data_output_path(request["cache_dir"])
                    if request.get("cache_dir")
                    else None,
                    max_laps=request.get("max_laps"),
                )
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return manifest

    @app.post("/data-sources/openf1/export")
    def openf1_export(payload: OpenF1ExportRequest) -> dict:
        request = _payload(payload)
        year = int(request["year"])
        event = str(request["event"])
        session = str(request["session"])
        driver = str(request["driver"])
        circuit_slug = event.strip().lower().replace(" ", "-")
        driver_slug = driver.strip().lower()
        session_slug = session.strip().lower()
        output = _safe_data_output_path(
            f"data/openf1-{year}-{circuit_slug}-{session_slug}-{driver_slug}.csv"
        )
        try:
            manifest = export_openf1_session(
                OpenF1ExportConfig(year=year, event=event, session=session, driver=driver, output=output)
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return manifest

    @app.get("/evaluation/replay-suite")
    def replay_suite() -> dict:
        try:
            report = run_replay_suite()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return replay_suite_to_dict(report)

    @app.get("/evaluation/replay-benchmark")
    def replay_benchmark() -> dict:
        try:
            report = run_benchmark_replay_suite()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return replay_suite_to_dict(report)

    @app.post("/evaluation/replay/run")
    def replay_run(payload: ReplayRunRequest) -> dict:
        kind = payload.kind.strip().lower()
        try:
            if kind == "dataset":
                report = run_replay_evaluation(_safe_dataset_path(payload.dataset_path), engine=engine)
                return {"kind": kind, "report": replay_report_to_dict(report)}
            if kind == "suite":
                report = run_replay_suite()
                return {"kind": kind, "suite": replay_suite_to_dict(report)}
            if kind == "benchmark":
                report = run_benchmark_replay_suite()
                return {"kind": kind, "suite": replay_suite_to_dict(report)}
            raise HTTPException(status_code=400, detail="kind must be one of dataset, suite, benchmark")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/regression/run")
    def regression_run(payload: RegressionRunRequest) -> dict:
        try:
            suite = RegressionSuite(
                config=RegressionConfig(
                    laps=payload.laps,
                    seed=payload.seed,
                    target_latency_ms=payload.target_latency_ms,
                    max_temporal_oscillation_s=payload.max_temporal_oscillation_s,
                    min_calibration_width_s=payload.min_calibration_width_s,
                    max_calibration_width_s=payload.max_calibration_width_s,
                )
            )
            results = suite.run()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "passed": all(result.passed for result in results),
            "results": [
                {
                    "name": result.name,
                    "passed": result.passed,
                    "value": result.value,
                    "threshold": result.threshold,
                }
                for result in results
            ],
            "config": {
                "laps": payload.laps,
                "seed": payload.seed,
                "target_latency_ms": payload.target_latency_ms,
                "max_temporal_oscillation_s": payload.max_temporal_oscillation_s,
                "min_calibration_width_s": payload.min_calibration_width_s,
                "max_calibration_width_s": payload.max_calibration_width_s,
            },
        }

    @app.post("/training/run")
    def training_run(payload: TrainingRequest) -> dict:
        valid_backends = ["xgboost", "lightgbm", "catboost", "sequence"]
        if payload.backend not in valid_backends:
            raise HTTPException(status_code=400, detail=f"backend must be one of {valid_backends}")
        real_data_paths: list[str] | None = None
        if payload.real_data:
            try:
                real_data_paths = [str(_safe_dataset_path(p)) for p in payload.real_data]
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        job_id = _uuid.uuid4().hex[:8]
        _training_jobs[job_id] = {
            "status": "running",
            "backend": payload.backend,
            "started_at": _time(),
            "log": [],
        }

        def _run() -> None:
            from f1_strategy.training import (
                _default_output,
                _serving_model_for_artifact,
                _training_config,
                train_model,
            )
            from f1_strategy.artifacts import create_model_artifact_bundle
            from f1_strategy.evaluation import run_evaluation
            from f1_strategy.replay import run_benchmark_replay_suite, run_replay_evaluation
            job = _training_jobs[job_id]
            try:
                output = _default_output(payload.backend)
                file_count = len(real_data_paths) if real_data_paths else 0
                job["log"].append(
                    f"Training {payload.backend} ({payload.seeds} seeds, {payload.rounds} rounds)"
                    + (f" + {file_count} real-data file(s)" if file_count else "")
                )
                output_path = train_model(
                    backend=payload.backend,
                    output=output,
                    laps=payload.laps,
                    seeds=payload.seeds,
                    rounds=payload.rounds,
                    real_data_paths=real_data_paths,
                    use_mlflow=payload.use_mlflow,
                    base_lap_time_s=payload.base_lap_time_s,
                )
                job["output_path"] = str(output_path)
                job["log"].append(f"Model saved: {output_path}")
                if payload.register_artifact:
                    artifact_root = engine.settings.model_artifact_root
                    job["log"].append("Evaluating artifact bundle…")
                    training_cfg = _training_config(
                        payload.backend, output_path, payload.laps, payload.seeds, payload.rounds,
                        real_data_paths=real_data_paths,
                    )
                    report = run_evaluation(
                        model_backend=payload.backend,
                        model_paths={payload.backend: str(output_path), "sequence": str(output_path)},
                    )
                    replay_report = run_replay_evaluation(
                        str(DEFAULT_REPLAY_DATASET),
                        engine=InferenceEngine(
                            model=_serving_model_for_artifact(payload.backend, output_path)
                        ),
                    )
                    replay_suite = run_benchmark_replay_suite(
                        engine_factory=lambda: InferenceEngine(
                            model=_serving_model_for_artifact(payload.backend, output_path)
                        )
                    )
                    bundle = create_model_artifact_bundle(
                        model_path=output_path,
                        backend=payload.backend,
                        training_config=training_cfg,
                        evaluation_report=report,
                        replay_evaluation_report=replay_report,
                        replay_suite_report=replay_suite,
                        artifact_root=artifact_root,
                    )
                    job["artifact_id"] = bundle.artifact_id
                    job["log"].append(f"Artifact registered: {bundle.artifact_id}")
                job["status"] = "done"
                job["completed_at"] = _time()
            except Exception as exc:
                job["status"] = "error"
                job["error"] = str(exc)
                job["completed_at"] = _time()

        threading.Thread(target=_run, daemon=True).start()
        return {"job_id": job_id, "status": "running", "backend": payload.backend}

    @app.get("/training/jobs")
    def training_jobs_list() -> dict:
        return {
            "jobs": [
                {
                    "job_id": k,
                    "status": v.get("status"),
                    "backend": v.get("backend"),
                    "started_at": v.get("started_at"),
                    "completed_at": v.get("completed_at"),
                    "artifact_id": v.get("artifact_id"),
                    "error": v.get("error"),
                    "log": v.get("log", []),
                }
                for k, v in sorted(
                    _training_jobs.items(), key=lambda x: x[1].get("started_at", 0), reverse=True
                )
            ]
        }

    @app.get("/training/status/{job_id}")
    def training_status_check(job_id: str) -> dict:
        job = _training_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}")
        return {"job_id": job_id, **job}

    @app.post("/shadow/promote")
    def shadow_promote() -> dict:
        global engine
        candidate = engine.shadow.promotion_candidate()
        if candidate is None:
            raise HTTPException(status_code=400, detail="No promotion candidate ready")
        challenger_backend = candidate["challenger_backend"]
        challenger_artifact = candidate.get("challenger_artifact", "unregistered")
        old_engine = engine
        try:
            if challenger_artifact and challenger_artifact != "unregistered":
                engine = InferenceEngine(
                    settings=replace(
                        settings, model_backend="auto", model_artifact_id=challenger_artifact
                    ),
                    monitoring=old_engine.monitoring,
                )
            else:
                normalized = challenger_backend.strip().lower()
                engine = InferenceEngine(
                    settings=replace(settings, model_backend=normalized, model_artifact_id=""),
                    monitoring=old_engine.monitoring,
                )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if hasattr(old_engine.persistence, "close"):
            old_engine.persistence.close()
        old_engine.shadow.disable()
        simulation.reset()
        return {
            **candidate,
            "promoted": True,
            "active_backend": getattr(
                engine.model, "backend_name", engine.model.__class__.__name__
            ),
            "active_artifact_id": engine.settings.model_artifact_id or "unregistered",
        }

    @app.get("/prediction/{session_id}/{car_id}")
    def prediction(session_id: str, car_id: str) -> dict:
        try:
            return to_jsonable(engine.predict(session_id, car_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/strategy/{session_id}/{car_id}")
    def strategy(session_id: str, car_id: str, remaining_laps: int = 30) -> dict:
        try:
            return to_jsonable(engine.strategy(session_id, car_id, remaining_laps))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

except ImportError:
    app = None


def _safe_dataset_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Invalid replay dataset path: {path}")
    allowed_roots = (Path("examples"), Path("data"))
    if not any(candidate == root or root in candidate.parents for root in allowed_roots):
        raise ValueError(f"Replay dataset must be under examples/ or data/: {path}")
    if candidate.suffix.lower() not in {".csv", ".jsonl"}:
        raise ValueError(f"Unsupported replay dataset type: {path}")
    if not candidate.exists():
        raise FileNotFoundError(f"Replay dataset not found: {candidate}")
    return candidate


def _safe_data_output_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Invalid data output path: {path}")
    if not (candidate == Path("data") or Path("data") in candidate.parents):
        raise ValueError(f"Data output path must be under data/: {path}")
    return candidate


def _default_fastf1_output(payload: dict) -> str:
    parts = [
        "fastf1",
        str(payload["year"]),
        _slug(payload["event"]),
        _slug(payload["session"]),
        _slug(payload["driver"]),
    ]
    return f"data/{'-'.join(parts)}.csv"


def _slug(value: object) -> str:
    return "".join(
        character.lower() if character.isalnum() else "-"
        for character in str(value).strip()
    ).strip("-")


def _list_replay_datasets() -> list[dict]:
    paths = sorted(
        [
            *Path("examples").glob("**/*.csv"),
            *Path("examples").glob("**/*.jsonl"),
            *Path("data").glob("**/*.csv"),
            *Path("data").glob("**/*.jsonl"),
        ]
    )
    return [_dataset_summary(path) for path in paths if path.is_file()]


def _dataset_summary(path: Path) -> dict:
    manifest_path = manifest_path_for(path)
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    event_count, labeled_count = _dataset_counts(path)
    fingerprint = manifest.get("dataset_fingerprint") if manifest else ""
    provenance = replay_data_provenance(path)
    return {
        "path": str(path),
        "name": path.stem,
        "source": manifest.get("source", "replay") if manifest else "replay",
        "dataset_fingerprint": fingerprint,
        "event_count": manifest.get("row_count", event_count) if manifest else event_count,
        "labeled_event_count": labeled_count,
        "lap_count": manifest.get("lap_count") if manifest else None,
        "has_manifest": manifest is not None,
        "manifest_path": str(manifest_path) if manifest_path.exists() else "",
        "field_provenance": manifest.get("field_provenance", {}) if manifest else {},
        "data_provenance": provenance,
        "validation_signal": provenance.get("validation_signal"),
        "production_validation_ready": provenance.get("production_validation_ready", False),
    }


def _dataset_counts(path: Path) -> tuple[int, int]:
    if path.suffix.lower() == ".jsonl":
        count = 0
        labeled = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                count += 1
                payload = json.loads(line)
                if payload.get("lap_time_s") is not None:
                    labeled += 1
        return count, labeled
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return len(rows), sum(1 for row in rows if row.get("lap_time_s"))
