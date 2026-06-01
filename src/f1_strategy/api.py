from __future__ import annotations

from dataclasses import replace

from f1_strategy.config import load_settings
from f1_strategy.deployment import load_registry
from f1_strategy.engine import InferenceEngine
from f1_strategy.live import LiveSimulationManager
from f1_strategy.metadata import APP_VERSION, build_info
from f1_strategy.models import FEATURE_SCHEMA_VERSION, feature_schema_hash
from f1_strategy.monitoring import monitoring_catalog
from f1_strategy.serialization import telemetry_from_dict, to_jsonable

engine = InferenceEngine()
settings = load_settings()
simulation = LiveSimulationManager()
runtime_build = build_info()
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
    from pathlib import Path

    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    app = FastAPI(title="F1 Tire and Energy Strategy API", version=APP_VERSION)
    ui_dir = Path(__file__).with_name("ui")
    app.mount("/ui", StaticFiles(directory=ui_dir), name="ui")

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

    class StatusResponse(BaseModel):
        status: str

    def _payload(model: BaseModel) -> dict:
        if hasattr(model, "model_dump"):
            return model.model_dump(exclude_none=True)
        return model.dict(exclude_none=True)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(ui_dir / "index.html")

    @app.get("/health")
    def health() -> dict[str, float | int | str]:
        return {
            "status": "ok",
            "version": runtime_build.version,
            "build_sha": runtime_build.build_sha,
            "build_date": runtime_build.build_date,
            "env": settings.env,
            "model_backend": getattr(engine.model, "backend_name", engine.model.__class__.__name__),
            "model_artifact_id": engine.settings.model_artifact_id or "unregistered",
            "persistence_backend": getattr(engine.persistence, "backend_name", "unknown"),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema_hash": feature_schema_hash(),
            "latency_p95_ms": round(engine.latency_p95_ms(), 4),
            "target_latency_ms": settings.target_latency_ms,
            "feature_window_size": settings.feature_window_size,
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
    def deployment_readiness() -> dict:
        return engine.deployment_readiness()

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
