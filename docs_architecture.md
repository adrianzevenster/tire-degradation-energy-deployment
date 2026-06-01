# Architecture Notes

## Runtime Flow

1. `TelemetryEvent` records arrive from a stream.
2. `OnlineFeatureStore.ingest` updates rolling state and derived features.
3. `InferenceEngine.ingest` produces a probabilistic forecast.
4. `StrategyOptimizer.recommend` converts forecasts into pit and ERS actions.
5. `DriftDetector.detect` compares current online features against a fitted baseline.
6. `RegressionSuite` validates latency, temporal stability, uncertainty width, and monotonic wear.

## Production Extension Points

- Replace `OnlineFeatureStore` with Redis, Feast, or a Kafka Streams materialized view.
- Use `F1_MODEL_BACKEND=xgboost` with a trained XGBoost artifact for production tabular serving.
- Use `F1_MODEL_BACKEND=lightgbm` or `catboost` for alternative tabular gradient boosting artifacts.
- Use `F1_MODEL_BACKEND=lstm`, `tft`, or `sequence` for TorchScript sequence artifacts.
- Use `F1_MODEL_BACKEND=kalman` for dependency-free online filtering or `river` for optional online learning.
- Replace `RaceSimulator` with replayed telemetry, live Kafka consumers, or a scenario generator.
- Add Prometheus counters around `InferenceEngine.latency_p95_ms`, drift alerts, and strategy recommendations.
- Persist predictions and features into ClickHouse or DuckDB for offline evaluation.

## Regression Gates

The bundled regression suite checks:

- p95 inference latency stays below the SLA.
- next-lap deltas do not oscillate beyond a stability threshold.
- uncertainty intervals remain within a calibrated operating band.
- tire wear does not decrease during a continuous stint.

These gates are intentionally simple and deterministic so they can run in CI before heavier model-training jobs.
