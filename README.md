# Real-Time Tire Degradation and Energy Deployment Prediction

This repository implements a production-style traditional ML system for Formula One strategy:

- streaming telemetry ingestion primitives
- online feature aggregation and session state
- tire degradation, brake temperature, ERS efficiency, and lap delta prediction
- pit-window and energy deployment optimization
- drift detection
- deterministic race simulation
- regression checks for accuracy, latency, calibration, stability, and drift

The first implementation is dependency-light and runnable with the Python standard library. Optional FastAPI, Prometheus, and ML dependencies can be added later without changing the core domain interfaces.

## Quick Start

```bash
python -m f1_strategy.cli --laps 8
python -m f1_strategy.cli --version
python -m f1_strategy.evaluation --format markdown
python -m f1_strategy.replay --dataset examples/replay_telemetry.csv
python -m unittest discover -s tests
```

If installed as a package:

```bash
pip install -e .
f1-simulate --laps 12
f1-regression
f1-evaluate --format json --output reports/model-evaluation.json
f1-replay-evaluate --dataset examples/replay_telemetry.csv
```

Common workflows are also available through `make`:

```bash
make test
make regression
make replay-evaluate
make reports
make ci
make train MODEL_BACKEND=xgboost MODEL_OUTPUT=models/xgboost_lap_delta.json
make train-evaluate MODEL_BACKEND=xgboost MODEL_OUTPUT=models/xgboost_lap_delta.json
make register-local-artifacts
make prune-artifacts
make promote-artifact ARTIFACT_ID=xgboost/2026-05-31T120102Z-abc1234
```

## Architecture

```text
Telemetry Stream
    -> Feature Aggregation Layer
    -> Online Feature Store
    -> Inference Engine
    -> Prediction API
    -> Strategy Optimization Engine
    -> Drift and Regression Monitors
```

## Models And Feature Store

The serving layer supports these model backends:

- `xgboost`: loads an XGBoost model artifact from `F1_XGBOOST_MODEL_PATH` and serves it
  behind the same `InferenceEngine` API.
- `lightgbm`: loads a LightGBM model artifact from `F1_LIGHTGBM_MODEL_PATH`.
- `catboost`: loads a CatBoost model artifact from `F1_CATBOOST_MODEL_PATH`.
- `lstm`, `tft`, or `sequence`: loads a TorchScript sequence model from
  `F1_SEQUENCE_MODEL_PATH`.
- `kalman`: uses a dependency-free online Kalman filter for pace smoothing.
- `river`: uses a River online regressor when the optional River dependency is installed.
- `hybrid`: uses `HybridOnlineEnsembleModel`, a dependency-light physics-informed
  online ensemble.

The default `F1_MODEL_BACKEND=auto` tries artifact-backed models first when their
dependencies and artifacts are available, then falls back to the hybrid model so
local/demo startup stays reliable.

`HybridOnlineEnsembleModel` combines domain physics priors for tire and thermal
behavior with an online ensemble of ridge residual learners. At serving time, each
telemetry event is converted into online features, the ensemble updates from observed
lap-time signal, and the model returns tire wear, tire life, grip loss, cliff
probability, brake temperature, ERS efficiency, lap delta, and uncertainty bounds.

Uncertainty is estimated from physics-prior uncertainty, ensemble disagreement, and a
rolling conformal residual buffer. This gives the service adaptive, session-specific
pace correction without adding heavyweight runtime dependencies.

Train and serve a tabular artifact with:

```bash
pip install -e ".[ml]"
python -m f1_strategy.training --backend xgboost --output models/xgboost_lap_delta.json
F1_MODEL_BACKEND=xgboost \
F1_XGBOOST_MODEL_PATH=models/xgboost_lap_delta.json \
python -m f1_strategy.cli --laps 8

python -m f1_strategy.training --backend lightgbm --output models/lightgbm_lap_delta.txt
F1_MODEL_BACKEND=lightgbm \
F1_LIGHTGBM_MODEL_PATH=models/lightgbm_lap_delta.txt \
python -m f1_strategy.cli --laps 8
```

Training writes a sidecar manifest next to each artifact, for example
`models/xgboost_lap_delta.json.manifest.json`. The manifest records the backend,
training row count, feature names, feature schema version, and feature schema hash.
Serving validates the manifest when present and rejects artifacts trained against a
different feature contract. XGBoost artifacts also store the feature schema hash in
the booster attributes.

For release-style lineage, use `make train-evaluate`. It trains the model, evaluates
the resulting artifact, and writes an immutable local bundle under
`artifacts/models/<backend>/<timestamp>-<git_sha>/` with:

- the model artifact
- `manifest.json`
- `training_config.json`
- `evaluation.json`
- `evaluation.md`
- `replay_evaluation.json`

The bundle manifest records artifact ID, git SHA, training parameters, feature schema
hash, simulated data fingerprint, replay dataset fingerprint, simulator evaluation
metrics, and replay holdout metrics. The local `artifacts/models/registry.json`
index tracks candidate artifacts and promoted artifacts. Generated artifacts stay
out of Git by default.

Promotion is a separate gate-checked step. `make promote-artifact ARTIFACT_ID=...`
validates that the manifest is complete, the model file and evaluation report exist,
the feature schema hash matches serving code, mean MAE and coverage satisfy thresholds,
p95 latency is within budget, monotonic tire-wear violations are zero, and replay
holdout gates pass. Successful promotion updates both the artifact manifest and
`registry.json`.

If model files already exist under `models/` but are not in the artifact registry,
register them as versioned candidates with:

```bash
make register-local-artifacts
```

This scans the default XGBoost, LightGBM, CatBoost, and sequence model paths, runs
simulator and replay evaluation for each available file, writes immutable bundles
under `artifacts/models`, and updates `artifacts/models/registry.json`.

When `F1_MODEL_BACKEND=auto` and `F1_MODEL_ARTIFACT_ID` is unset, the service loads
the latest promoted artifact from the registry on startup. Archive older candidate
entries with:

```bash
make prune-artifacts
```

Serve a registered bundle by artifact ID:

```bash
F1_MODEL_ARTIFACT_ID=xgboost/2026-05-31T120102Z-abc1234 \
F1_MODEL_ARTIFACT_ROOT=artifacts/models \
python -m f1_strategy.cli --laps 8
```

CatBoost training uses the same trainer with `--backend catboost` after installing
`pip install -e ".[catboost]"`. River serving uses `pip install -e ".[online]"`.
TorchScript LSTM/TFT serving uses `pip install -e ".[deep]"`; a local sequence artifact
can be trained with:

```bash
python -m f1_strategy.training --backend sequence --output models/sequence_lap_delta.pt
F1_MODEL_BACKEND=tft \
F1_SEQUENCE_MODEL_PATH=models/sequence_lap_delta.pt \
python -m f1_strategy.cli --laps 8
```

The current feature store is `OnlineFeatureStore`, an in-memory online store keyed by
`session_id` and `car_id`. It maintains a rolling telemetry window and materializes
serving features such as tire age, mean tire temperature, brake heat index, driver
aggression, degradation acceleration, ERS efficiency, rolling lap time, and dirty-air
risk. This keeps local inference dependency-light, but it is not yet a durable
production feature platform. The store interface is intentionally small so it can be
backed later by Redis, Feast, ClickHouse, or a stream processor without changing the
API or inference engine contract.

DuckDB persistence is the default local durable storage when the optional
dependency is installed. `F1_PERSISTENCE_BACKEND=auto` tries DuckDB and falls
back to no-op persistence when DuckDB is unavailable, so demo startup continues
to work in dependency-light environments.

```bash
pip install -e ".[persistence]"
F1_PERSISTENCE_BACKEND=auto \
F1_DUCKDB_PATH=data/f1_strategy.duckdb \
python -m f1_strategy.cli --laps 8
```

When enabled, the engine records telemetry events, online features, predictions,
strategy recommendations, and evaluations into DuckDB tables. Stored feature and
prediction rows include the active feature schema version and hash so records can be
joined back to the model contract used at serving time.

The API exposes persisted run summaries through `GET /history/runs`, and the browser
UI uses that endpoint to compare recent simulation runs.

CI gates run the regression suite to protect latency, temporal stability, uncertainty
width, and monotonic tire-wear behavior.

## Evaluation Reports

Run deterministic scenario evaluation with:

```bash
python -m f1_strategy.evaluation --format markdown
python -m f1_strategy.evaluation --format json --output reports/model-evaluation.json
```

The evaluation report covers baseline medium, soft degradation, hard long-run, and
intermediate-pace scenarios. It reports lap-delta MAE/RMSE, interval coverage,
average prediction interval width, p95 serving latency, and monotonic tire-wear
violations. Reports also include the active feature schema version and hash. This
complements the CI regression suite by producing a portable model quality artifact
for reviews, releases, and model comparisons.

Replay evaluation is the production-facing complement to simulator regression. A
replay dataset can be CSV or JSONL with the same telemetry fields accepted by
`POST /telemetry`: `session_id`, `car_id`, `lap`, `sector`, speed/control signals,
tire/brake temperatures, ERS/fuel/weather fields, `compound`, and optional
`timestamp_ms`. Include `lap_time_s` for every holdout row when the dataset should
act as a promotion gate. Run:

```bash
python -m f1_strategy.replay --dataset examples/replay_telemetry.csv
```

The replay report records the dataset fingerprint, session/event counts, labeled
row count, target completeness, MAE/RMSE, coverage, p95 latency, monotonic wear
violations, and pass/fail gates. The API exposes the same report at
`GET /evaluation/replay`.

## API Service

The core service can run as FastAPI when the optional API dependencies are installed:

```bash
pip install -e ".[api]"
uvicorn f1_strategy.api:app --reload
```

Endpoints:

- `POST /telemetry`: ingest one telemetry event
- `GET /prediction/{session_id}/{car_id}`: return current forecast
- `GET /strategy/{session_id}/{car_id}`: return optimized pit and ERS strategy
- `POST /evaluation`: record actual lap outcome metrics against the latest prediction
- `POST /simulation/start`: start a live simulation run
- `POST /simulation/stop`: pause the live simulation
- `POST /simulation/reset`: clear the current simulation state
- `POST /simulation/tick`: advance one or more telemetry events
- `GET /simulation/status`: return live simulation progress
- `GET /history/runs`: return persisted run summaries for comparison
- `GET /monitoring/model-performance`: return rolling model performance by artifact
- `GET /monitoring/alerts`: return active model performance, latency, and drift alerts
- `GET /deployment/readiness`: return deployment readiness checks and rollback candidate
- `GET /deployment/rollback-candidate`: return the selected rollback artifact candidate
- `GET /metrics`: Prometheus-compatible metric exposition
- `GET /monitoring/catalog`: metric names grouped by ML, infrastructure, and racing semantics
- `GET /health`: basic service health
- `GET /version`: package version and build metadata

The API uses typed request models for telemetry and evaluation payloads. Invalid
payloads, such as out-of-range sector, throttle, brake, ERS state of charge, or
humidity values, are rejected at the HTTP boundary with FastAPI validation errors.
`GET /health` includes the active model backend, feature schema version, and feature
schema hash for deployment inspection.

## Environment And Monitoring

Copy `.env.example` to `.env` for local overrides. The main knobs are:

- `F1_TARGET_LATENCY_MS`
- `F1_FEATURE_WINDOW_SIZE`
- `F1_BUILD_SHA`
- `F1_BUILD_DATE`
- `F1_MODEL_BACKEND`
- `F1_MODEL_ARTIFACT_ID`
- `F1_MODEL_ARTIFACT_ROOT`
- `F1_XGBOOST_MODEL_PATH`
- `F1_LIGHTGBM_MODEL_PATH`
- `F1_CATBOOST_MODEL_PATH`
- `F1_SEQUENCE_MODEL_PATH`
- `F1_DRIFT_THRESHOLD_Z`
- `F1_MAX_TEMPORAL_OSCILLATION_S`
- `F1_MAX_CALIBRATION_WIDTH_S`
- `F1_PERSISTENCE_BACKEND`
- `F1_DUCKDB_PATH`

Run the observability stack with:

```bash
docker compose up --build
```

For traceable images, pass build metadata through Docker build args:

```bash
docker build \
  --build-arg F1_BUILD_SHA="$(git rev-parse --short HEAD)" \
  --build-arg F1_BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t f1-tire-energy-strategy:0.1.0 .
```

Services:

- API: http://localhost:8000
- Live UI: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000, login `admin` / `admin`

Metrics cover:

- ML: RMSE, MAE, MAPE, calibration error, prediction interval width, tire wear, cliff probability
- Infrastructure: inference latency, throughput, queue depth, drift alerts
- Racing: pit recommendation count/accuracy, tire-cliff accuracy, ERS efficiency, battery depletion error, lap delta

## Live UI

Install and run the API service:

```bash
pip install -e ".[api]"
uvicorn f1_strategy.api:app --reload
```

Open `http://localhost:8000`. The UI can start a synthetic race stream, advance telemetry batches, and view live predictions, strategy windows, ERS deployment, drift scores, latency, and Prometheus metrics.

## Notes

The bundled model stack supports XGBoost, LightGBM, CatBoost, TorchScript LSTM/TFT,
Kalman, and River-style online serving behind the same `InferenceEngine` contract,
with a dependency-light online ensemble fallback.
