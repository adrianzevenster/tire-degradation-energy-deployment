from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from f1_strategy.config import load_settings
from f1_strategy.domain import TireCompound
from f1_strategy.drift import DriftDetector
from f1_strategy.engine import InferenceEngine
from f1_strategy.artifacts import (
    PromotionGateConfig,
    artifact_release_detail,
    create_model_artifact_bundle,
    promote_artifact,
    prune_artifact_registry,
    resolve_model_artifact,
)
from f1_strategy.deployment import load_registry, rollback_candidate
from f1_strategy.evaluation import (
    EvaluationReport,
    ScenarioEvaluation,
    render_markdown,
    report_to_dict,
    run_evaluation,
)
from f1_strategy.live import LiveSimulationManager
from f1_strategy.metadata import APP_VERSION, build_info
from f1_strategy.models import (
    CatBoostModel,
    HybridOnlineEnsembleModel,
    KalmanFilterModel,
    LightGBMModel,
    ModelConfig,
    RiverOnlineModel,
    SequenceTorchModel,
    XGBoostModel,
    create_serving_model,
    feature_schema_hash,
    model_manifest_path,
    validate_model_manifest,
    write_model_manifest,
)
from f1_strategy.monitoring import MonitoringService, monitoring_catalog
from f1_strategy.persistence import (
    DuckDBPersistence,
    InMemoryPersistence,
    NullPersistence,
    create_persistence_store,
)
from f1_strategy.regression import RegressionSuite
from f1_strategy.replay import (
    REPLAY_REQUIRED_COLUMNS,
    ReplayEvaluationReport,
    ReplaySuiteReport,
    load_replay_events,
    replay_report_to_dict,
    replay_suite_to_dict,
    run_replay_evaluation,
    run_replay_suite,
)
from f1_strategy.serialization import telemetry_from_dict, to_jsonable
from f1_strategy.simulation import RaceSimulator, SimulationConfig
from f1_strategy.training import train_xgboost_model


class RecordingPersistence:
    backend_name = "recording"

    def __init__(self) -> None:
        self.telemetry = 0
        self.features = 0
        self.predictions = 0
        self.strategies = 0
        self.evaluations = 0

    def record_telemetry(self, event) -> None:
        self.telemetry += 1

    def record_features(self, features) -> None:
        self.features += 1

    def record_prediction(self, prediction, latency_ms: float) -> None:
        self.predictions += 1

    def record_strategy(self, recommendation) -> None:
        self.strategies += 1

    def record_evaluation(
        self,
        session_id: str,
        car_id: str,
        actual_lap_delta_s: float,
        prediction,
        actual_cliff: bool | None = None,
        actual_ending_soc: float | None = None,
    ) -> None:
        self.evaluations += 1


class SystemTest(unittest.TestCase):
    def test_streaming_inference_and_strategy(self) -> None:
        engine = InferenceEngine()
        self.assertIsInstance(
            engine.model,
            (
                HybridOnlineEnsembleModel,
                XGBoostModel,
                LightGBMModel,
                CatBoostModel,
                SequenceTorchModel,
                KalmanFilterModel,
                RiverOnlineModel,
            ),
        )
        simulator = RaceSimulator(SimulationConfig(laps=6, seed=3))
        prediction = None
        for event in simulator.events():
            prediction = engine.ingest(event)

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertGreaterEqual(prediction.tire_wear_pct, 0.0)
        self.assertLessEqual(prediction.tire_wear_pct, 100.0)
        self.assertLess(
            prediction.uncertainty_low_s,
            prediction.uncertainty_high_s,
        )

        strategy = engine.strategy(prediction.session_id, prediction.car_id, remaining_laps=30)
        self.assertGreaterEqual(strategy.pit_window.target_lap, prediction.lap + 1)
        self.assertEqual(set(strategy.energy_plan.sector_deployment_kw), {1, 2, 3})
        observations = getattr(
            engine.model,
            "observations",
            getattr(getattr(engine.model, "fallback", None), "observations", 0),
        )
        self.assertGreater(observations, 0)

    def test_tire_wear_is_monotonic_over_stint(self) -> None:
        engine = InferenceEngine()
        simulator = RaceSimulator(SimulationConfig(laps=10, seed=9, compound=TireCompound.SOFT))
        lap_wear = []
        for index, event in enumerate(simulator.events(), start=1):
            prediction = engine.ingest(event)
            if index % 3 == 0:
                lap_wear.append(prediction.tire_wear_pct)

        self.assertEqual(lap_wear, sorted(lap_wear))

    def test_drift_detection_flags_shifted_track_temperature(self) -> None:
        engine = InferenceEngine()
        baseline_events = RaceSimulator(SimulationConfig(laps=8, seed=4)).events()
        for event in baseline_events:
            engine.ingest(event)
        baseline = engine.feature_store.snapshot()

        detector = DriftDetector(threshold_z=2.0)
        detector.fit_baseline(baseline)
        engine.drift_detector = detector

        hot_events = RaceSimulator(SimulationConfig(laps=1, seed=5)).events()
        hot_event = hot_events[-1]
        shifted = hot_event.__class__(
            **{**hot_event.__dict__, "session_id": "hot", "track_temp_c": 55.0}
        )
        engine.ingest(shifted)
        report = engine.drift("hot", shifted.car_id)
        self.assertTrue(report.drifted)
        self.assertTrue(any("track_temp_c" in alert for alert in report.alerts))

    def test_serialization_round_trip(self) -> None:
        event = RaceSimulator(SimulationConfig(laps=1)).events()[0]
        payload = to_jsonable(event)
        parsed = telemetry_from_dict(payload)
        self.assertEqual(parsed.compound, event.compound)
        self.assertEqual(parsed.session_id, event.session_id)

    def test_regression_suite_passes(self) -> None:
        results = RegressionSuite().run()
        self.assertTrue(all(result.passed for result in results), results)

    def test_evaluation_report_covers_default_scenarios(self) -> None:
        report = run_evaluation()
        self.assertGreaterEqual(len(report.scenarios), 3)
        self.assertGreater(report.mean_coverage_pct, 0.0)
        self.assertTrue(all(item.observations > 0 for item in report.scenarios))
        self.assertTrue(all(item.monotonic_wear_violations == 0 for item in report.scenarios))

        markdown = render_markdown(report)
        self.assertIn("Model Evaluation Report", markdown)
        self.assertIn("medium-baseline", markdown)
        payload = report_to_dict(report)
        self.assertEqual(payload["scenario_count"], len(report.scenarios))
        self.assertIn("mean_mae_lap_delta_s", payload)
        self.assertEqual(payload["feature_schema_hash"], feature_schema_hash())

    def test_replay_evaluation_reports_holdout_gates(self) -> None:
        events = load_replay_events("examples/replay_telemetry.csv")
        self.assertGreater(len(events), 0)
        self.assertTrue(all(getattr(events[0], name) is not None for name in REPLAY_REQUIRED_COLUMNS))

        report = run_replay_evaluation("examples/replay_telemetry.csv")
        payload = replay_report_to_dict(report)

        self.assertEqual(report.event_count, len(events))
        self.assertEqual(report.labeled_event_count, len(events))
        self.assertEqual(report.missing_target_pct, 0.0)
        self.assertEqual(payload["scenario"]["source"], "replay")
        self.assertIn("dataset_fingerprint", payload)
        self.assertTrue(all(isinstance(value, bool) for value in report.gates.values()))
        self.assertTrue(report.passed, payload)

    def test_replay_suite_reports_named_splits(self) -> None:
        report = run_replay_suite()
        payload = replay_suite_to_dict(report)

        self.assertGreaterEqual(report.split_count, 5)
        self.assertTrue(report.passed, payload)
        self.assertIn("smoke", {split.scenario.scenario for split in report.splits})
        self.assertGreater(report.total_event_count, 12)
        self.assertTrue(all(split.dataset_fingerprint for split in report.splits))

    def test_model_manifest_validates_feature_schema_hash(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "model.json"
            artifact.write_text("{}", encoding="utf-8")
            manifest = write_model_manifest(artifact, backend="test", training_rows=12)

            self.assertEqual(manifest, model_manifest_path(artifact))
            validate_model_manifest(artifact, backend="test")

            manifest.write_text(
                '{"feature_schema_hash": "wrong", "feature_names": []}\n',
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                validate_model_manifest(artifact, backend="test")

    def test_model_artifact_bundle_records_lineage_and_registry(self) -> None:
        report = self._artifact_test_report()
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.json"
            model_path.write_text("{}", encoding="utf-8")
            bundle = create_model_artifact_bundle(
                model_path=model_path,
                backend="xgboost",
                training_config={
                    "backend": "xgboost",
                    "laps": 2,
                    "seeds": 3,
                    "rounds": 4,
                    "training_rows": 18,
                },
                evaluation_report=report,
                replay_evaluation_report=self._artifact_replay_report(),
                replay_suite_report=self._artifact_replay_suite_report(),
                artifact_root=Path(temp_dir) / "artifacts",
                created_at="2026-05-31T120102Z",
                git_sha="abc1234",
            )

            self.assertEqual(bundle.artifact_id, "xgboost/2026-05-31T120102Z-abc1234")
            self.assertTrue(bundle.model_path.exists())
            self.assertTrue(bundle.manifest_path.exists())
            self.assertTrue((bundle.bundle_dir / "evaluation.json").exists())
            self.assertTrue((bundle.bundle_dir / "evaluation.md").exists())
            self.assertTrue((bundle.bundle_dir / "replay_evaluation.json").exists())
            self.assertTrue((bundle.bundle_dir / "replay_suite.json").exists())
            manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["replay_dataset_fingerprint"], "replay-fingerprint")
            self.assertTrue(manifest["replay_evaluation_metrics"]["passed"])
            self.assertTrue(manifest["replay_suite_metrics"]["passed"])
            validate_model_manifest(bundle.model_path, backend="xgboost")
            backend, resolved_model_path = resolve_model_artifact(
                bundle.artifact_id,
                artifact_root=Path(temp_dir) / "artifacts",
            )
            self.assertEqual(backend, "xgboost")
            self.assertEqual(resolved_model_path, bundle.model_path)

    def test_promote_artifact_updates_manifest_and_registry_after_gates_pass(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bundle = self._create_test_artifact(temp_dir, report=self._artifact_test_report())

            result = promote_artifact(
                bundle.artifact_id,
                artifact_root=Path(temp_dir) / "artifacts",
                gates=PromotionGateConfig(
                    max_mean_mae_lap_delta_s=0.2,
                    min_mean_coverage_pct=95.0,
                    max_latency_p95_ms=2.0,
                ),
            )

            self.assertTrue(result.promoted)
            manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
            registry = json.loads(bundle.registry_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "promoted")
            self.assertEqual(registry["promoted"]["xgboost"], bundle.artifact_id)
            self.assertEqual(registry["artifacts"][0]["status"], "promoted")
            self.assertTrue(registry["artifacts"][0]["replay_passed"])
            self.assertEqual(registry["artifacts"][0]["replay_mae_lap_delta_s"], 0.1)

    def test_artifact_release_detail_exposes_blockers_and_reports(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bundle = self._create_test_artifact(temp_dir, report=self._artifact_test_report())

            detail = artifact_release_detail(
                bundle.artifact_id,
                artifact_root=Path(temp_dir) / "artifacts",
            )

            self.assertTrue(detail["promotion_ready"])
            self.assertEqual(detail["promotion_failures"], [])
            self.assertEqual(detail["manifest"]["artifact_id"], bundle.artifact_id)
            self.assertIsNotNone(detail["evaluation"])
            self.assertIsNotNone(detail["replay_evaluation"])
            self.assertIsNotNone(detail["replay_suite"])
            self.assertTrue(detail["replay_evaluation"]["passed"])

    def test_promote_artifact_rejects_failed_gates_without_registry_promotion(self) -> None:
        report = self._artifact_test_report(
            mae_lap_delta_s=2.0,
            coverage_pct=25.0,
            latency_p95_ms=50.0,
            monotonic_wear_violations=1,
        )
        with TemporaryDirectory() as temp_dir:
            bundle = self._create_test_artifact(temp_dir, report=report)

            result = promote_artifact(
                bundle.artifact_id,
                artifact_root=Path(temp_dir) / "artifacts",
                gates=PromotionGateConfig(
                    max_mean_mae_lap_delta_s=0.2,
                    min_mean_coverage_pct=95.0,
                    max_latency_p95_ms=2.0,
                ),
            )

            self.assertFalse(result.promoted)
            self.assertGreaterEqual(len(result.failures), 4)
            manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
            registry = json.loads(bundle.registry_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "candidate")
            self.assertEqual(registry["promoted"], {})
            self.assertEqual(registry["artifacts"][0]["status"], "candidate")

    def test_promote_artifact_rejects_missing_replay_evaluation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.json"
            model_path.write_text("{}", encoding="utf-8")
            bundle = create_model_artifact_bundle(
                model_path=model_path,
                backend="xgboost",
                training_config={
                    "backend": "xgboost",
                    "laps": 2,
                    "seeds": 3,
                    "rounds": 4,
                    "training_rows": 18,
                },
                evaluation_report=self._artifact_test_report(),
                artifact_root=Path(temp_dir) / "artifacts",
                created_at="2026-05-31T120102Z",
                git_sha="abc1234",
            )

            result = promote_artifact(
                bundle.artifact_id,
                artifact_root=Path(temp_dir) / "artifacts",
            )

            self.assertFalse(result.promoted)
            self.assertTrue(any("replay" in failure for failure in result.failures))

    def test_promote_artifact_rejects_failed_replay_evaluation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bundle = self._create_test_artifact(
                temp_dir,
                report=self._artifact_test_report(),
                replay_report=self._artifact_replay_report(
                    passed=False,
                    mae_lap_delta_s=1.2,
                    coverage_pct=20.0,
                    missing_target_pct=25.0,
                    monotonic_wear_violations=2,
                ),
                replay_suite_report=self._artifact_replay_suite_report(passed=False),
            )

            result = promote_artifact(
                bundle.artifact_id,
                artifact_root=Path(temp_dir) / "artifacts",
            )

            self.assertFalse(result.promoted)
            self.assertTrue(any("replay" in failure for failure in result.failures))

    def test_monitoring_catalog_and_export_cover_spec_metrics(self) -> None:
        catalog = monitoring_catalog()
        self.assertIn("rmse", catalog["ml"])
        self.assertIn("model_mae_lap_delta_s", catalog["ml"])
        self.assertIn("inference_latency_ms", catalog["infrastructure"])
        self.assertIn("pit_stop_recommendation_accuracy", catalog["racing"])

        engine = InferenceEngine()
        event = RaceSimulator(SimulationConfig(laps=1)).events()[-1]
        prediction = engine.ingest(event)
        engine.strategy(prediction.session_id, prediction.car_id, remaining_laps=20)
        engine.monitoring.record_evaluation(actual_lap_delta_s=0.2, prediction=prediction)
        metrics = engine.monitoring.render_prometheus()

        self.assertIn("f1_inference_latency_ms_p95", metrics)
        self.assertIn("f1_tire_wear_pct", metrics)
        self.assertIn("f1_pit_stop_recommendations_total", metrics)
        self.assertIn("f1_calibration_error", metrics)

    def test_model_performance_monitoring_groups_evaluations_by_identity(self) -> None:
        engine = InferenceEngine()
        event = RaceSimulator(SimulationConfig(laps=1, seed=17)).events()[-1]
        prediction = engine.ingest(event)

        self.assertNotEqual(prediction.model_backend, "unknown")
        self.assertNotEqual(prediction.model_artifact_id, "unknown")
        self.assertEqual(prediction.model_feature_schema_hash, feature_schema_hash())

        performance = engine.model_performance()
        metrics = engine.monitoring.render_prometheus()

        self.assertEqual(len(performance), 1)
        self.assertEqual(performance[0]["backend"], prediction.model_backend)
        self.assertEqual(performance[0]["artifact_id"], prediction.model_artifact_id)
        self.assertEqual(performance[0]["evaluations"], 1)
        self.assertGreaterEqual(performance[0]["mae_lap_delta_s"], 0.0)
        self.assertIn(f'f1_model_mae_lap_delta_s{{artifact_id="{prediction.model_artifact_id}"', metrics)
        self.assertIn("f1_model_rmse_lap_delta_s", metrics)
        self.assertIn("f1_model_interval_coverage_pct", metrics)
        self.assertIn("f1_model_evaluations_total", metrics)

    def test_model_comparison_ranks_multiple_evaluated_backends(self) -> None:
        monitoring = MonitoringService()
        hybrid = InferenceEngine(monitoring=monitoring)
        kalman = InferenceEngine(
            monitoring=monitoring,
            settings=hybrid.settings.__class__(model_backend="kalman"),
        )
        for engine, seed in ((hybrid, 21), (kalman, 22)):
            for event in RaceSimulator(SimulationConfig(laps=1, seed=seed)).events():
                engine.ingest(event)

        comparison = hybrid.model_comparison()

        self.assertGreaterEqual(len(comparison), 2)
        self.assertEqual([row["rank"] for row in comparison], list(range(1, len(comparison) + 1)))
        self.assertTrue(all(row["evaluations"] >= 1 for row in comparison))
        self.assertIn("mae_lap_delta_s", comparison[0])

    def test_model_alerts_flag_missing_labels_and_performance_regression(self) -> None:
        engine = InferenceEngine()
        missing_label_alerts = engine.model_alerts()
        self.assertTrue(
            any(alert["type"] == "missing_labels" for alert in missing_label_alerts["alerts"])
        )

        event = RaceSimulator(SimulationConfig(laps=1, seed=18)).events()[-1]
        prediction = engine.ingest(event)
        engine.record_evaluation(
            prediction.session_id,
            prediction.car_id,
            actual_lap_delta_s=prediction.next_lap_delta_s + 5.0,
            actual_cliff=not (prediction.cliff_probability >= 0.50),
        )

        alerts = engine.model_alerts()
        alert_types = {alert["type"] for alert in alerts["alerts"]}
        metrics = engine.monitoring.render_prometheus()
        self.assertIn("mae_lap_delta", alert_types)
        self.assertLess(alerts["health_score"], 100.0)
        self.assertIn("f1_model_alerts_total", metrics)
        self.assertIn("f1_model_health_score", metrics)

    def test_model_alerts_flag_repeated_drift(self) -> None:
        engine = InferenceEngine()
        baseline_events = RaceSimulator(SimulationConfig(laps=8, seed=19)).events()
        for event in baseline_events:
            engine.ingest(event)
        detector = DriftDetector(threshold_z=2.0)
        detector.fit_baseline(engine.feature_store.snapshot())
        engine.drift_detector = detector

        hot_event = RaceSimulator(SimulationConfig(laps=1, seed=20)).events()[-1]
        for index in range(2):
            shifted = hot_event.__class__(
                **{
                    **hot_event.__dict__,
                    "session_id": f"drift-{index}",
                    "track_temp_c": 58.0,
                }
            )
            engine.ingest(shifted)
            engine.drift(shifted.session_id, shifted.car_id)

        alerts = engine.model_alerts()
        self.assertTrue(any(alert["type"] == "drift_track_temp_c" for alert in alerts["alerts"]))

    def test_deployment_readiness_exposes_checks_and_prometheus_metric(self) -> None:
        engine = InferenceEngine()
        readiness = engine.deployment_readiness()
        metrics = engine.monitoring.render_prometheus()

        self.assertIn("ready", readiness)
        self.assertIn("checks", readiness)
        self.assertIn("rollback_candidate", readiness)
        self.assertIn("f1_deployment_ready", metrics)

    def test_rollback_candidate_selects_previous_promoted_artifact(self) -> None:
        report = self._artifact_test_report()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "artifacts"
            first = self._create_test_artifact(
                temp_dir,
                report=report,
                created_at="2026-05-31T120102Z",
                git_sha="aaa111",
            )
            second = self._create_test_artifact(
                temp_dir,
                report=report,
                created_at="2026-05-31T120202Z",
                git_sha="bbb222",
            )
            promote_artifact(first.artifact_id, artifact_root=root)
            promote_artifact(second.artifact_id, artifact_root=root)

            candidate = rollback_candidate(
                load_registry(root),
                backend="xgboost",
                active_artifact_id=second.artifact_id,
            )

            self.assertIsNotNone(candidate)
            assert candidate is not None
            self.assertEqual(candidate["artifact_id"], first.artifact_id)

    def test_engine_auto_loads_latest_promoted_artifact(self) -> None:
        try:
            import xgboost  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"XGBoost is not installed: {exc}")

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "artifacts"
            model_path = train_xgboost_model(
                str(Path(temp_dir) / "xgboost_lap_delta.json"),
                laps=1,
                seeds=1,
                rounds=1,
            )
            bundle = create_model_artifact_bundle(
                model_path=model_path,
                backend="xgboost",
                training_config={
                    "backend": "xgboost",
                    "laps": 1,
                    "seeds": 1,
                    "rounds": 1,
                    "training_rows": 1,
                },
                evaluation_report=self._artifact_test_report(),
                replay_evaluation_report=self._artifact_replay_report(),
                replay_suite_report=self._artifact_replay_suite_report(),
                artifact_root=root,
                created_at="2026-05-31T120402Z",
                git_sha="ddd444",
            )
            promote_artifact(bundle.artifact_id, artifact_root=root)

            settings = load_settings().__class__(
                model_backend="auto",
                model_artifact_id="",
                model_artifact_root=str(root),
            )
            engine = InferenceEngine(settings=settings)

            self.assertEqual(engine.settings.model_artifact_id, bundle.artifact_id)
            self.assertEqual(engine._active_model_artifact_id(), bundle.artifact_id)

    def test_prune_artifact_registry_archives_old_candidates(self) -> None:
        report = self._artifact_test_report()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "artifacts"
            old = self._create_test_artifact(
                temp_dir,
                report=report,
                created_at="2026-05-31T120001Z",
                git_sha="old111",
            )
            new = self._create_test_artifact(
                temp_dir,
                report=report,
                created_at="2026-05-31T120002Z",
                git_sha="new222",
            )

            result = prune_artifact_registry(root, keep_candidates_per_backend=1)
            registry = load_registry(root)
            statuses = {item["artifact_id"]: item["status"] for item in registry["artifacts"]}

            self.assertIn(old.artifact_id, result.archived)
            self.assertIn(new.artifact_id, result.kept)
            self.assertEqual(statuses[old.artifact_id], "archived")
            self.assertEqual(statuses[new.artifact_id], "candidate")

    def test_artifact_id_loaded_model_reports_registered_identity(self) -> None:
        try:
            import xgboost  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"XGBoost is not installed: {exc}")

        report = self._artifact_test_report()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "artifacts"
            model_path = train_xgboost_model(
                str(Path(temp_dir) / "xgboost_lap_delta.json"),
                laps=1,
                seeds=1,
                rounds=1,
            )
            bundle = create_model_artifact_bundle(
                model_path=model_path,
                backend="xgboost",
                training_config={
                    "backend": "xgboost",
                    "laps": 1,
                    "seeds": 1,
                    "rounds": 1,
                    "training_rows": 1,
                },
                evaluation_report=report,
                replay_evaluation_report=self._artifact_replay_report(),
                replay_suite_report=self._artifact_replay_suite_report(),
                artifact_root=root,
                created_at="2026-05-31T120302Z",
                git_sha="ccc333",
            )
            promote_artifact(bundle.artifact_id, artifact_root=root)
            settings = load_settings().__class__(
                model_artifact_id=bundle.artifact_id,
                model_artifact_root=str(root),
            )
            engine = InferenceEngine(settings=settings)
            event = RaceSimulator(SimulationConfig(laps=1, seed=24)).events()[0]
            prediction = engine.ingest(event)

            self.assertEqual(prediction.model_artifact_id, bundle.artifact_id)
            self.assertEqual(engine.deployment_readiness()["promoted"], True)

    def test_engine_records_persistence_events(self) -> None:
        persistence = RecordingPersistence()
        engine = InferenceEngine(persistence=persistence)
        event = RaceSimulator(SimulationConfig(laps=1, seed=10)).events()[-1]
        prediction = engine.ingest(event)
        engine.strategy(prediction.session_id, prediction.car_id, remaining_laps=20)
        engine.record_evaluation(
            prediction.session_id,
            prediction.car_id,
            actual_lap_delta_s=0.2,
        )

        self.assertEqual(persistence.telemetry, 1)
        self.assertEqual(persistence.features, 1)
        self.assertGreaterEqual(persistence.predictions, 2)
        self.assertEqual(persistence.strategies, 1)
        self.assertEqual(persistence.evaluations, 2)

    def test_null_persistence_factory(self) -> None:
        self.assertIsInstance(create_persistence_store("none", "ignored.duckdb"), NullPersistence)
        with TemporaryDirectory() as temp_dir:
            auto_store = create_persistence_store("auto", str(Path(temp_dir) / "state.duckdb"))
            self.assertIn(auto_store.backend_name, {"memory", "duckdb"})
            if hasattr(auto_store, "close"):
                auto_store.close()
        self.assertEqual(create_persistence_store("none", "ignored.duckdb").run_summaries(), [])

    def test_memory_persistence_lists_run_summaries(self) -> None:
        persistence = InMemoryPersistence()
        engine = InferenceEngine(persistence=persistence)
        simulator = RaceSimulator(SimulationConfig(session_id="sim-race-memory", laps=2, seed=14))
        for event in simulator.events():
            prediction = engine.ingest(event)
        engine.strategy(prediction.session_id, prediction.car_id, remaining_laps=20)

        summaries = persistence.run_summaries()

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["session_id"], "sim-race-memory")
        self.assertEqual(summaries[0]["prediction_count"], 6)
        self.assertIsNotNone(summaries[0]["latest_strategy"])

    def test_duckdb_persistence_records_rows_when_available(self) -> None:
        try:
            import duckdb  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"DuckDB is not installed: {exc}")

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.duckdb"
            persistence = DuckDBPersistence(path)
            engine = InferenceEngine(persistence=persistence)
            event = RaceSimulator(SimulationConfig(laps=1, seed=13)).events()[-1]
            prediction = engine.ingest(event)
            engine.strategy(prediction.session_id, prediction.car_id, remaining_laps=20)
            engine.record_evaluation(
                prediction.session_id,
                prediction.car_id,
                actual_lap_delta_s=0.2,
            )

            telemetry_count = persistence.connection.execute(
                "SELECT count(*) FROM telemetry_events"
            ).fetchone()[0]
            prediction_count = persistence.connection.execute(
                "SELECT count(*) FROM predictions"
            ).fetchone()[0]
            evaluation_count = persistence.connection.execute(
                "SELECT count(*) FROM evaluations"
            ).fetchone()[0]
            prediction_identity = persistence.connection.execute(
                "SELECT model_backend, model_artifact_id FROM predictions LIMIT 1"
            ).fetchone()
            evaluation_identity = persistence.connection.execute(
                "SELECT model_backend, model_artifact_id FROM evaluations LIMIT 1"
            ).fetchone()
            persistence.close()

        self.assertEqual(telemetry_count, 1)
        self.assertGreaterEqual(prediction_count, 2)
        self.assertEqual(evaluation_count, 2)
        self.assertNotEqual(prediction_identity[0], "unknown")
        self.assertNotEqual(prediction_identity[1], "unknown")
        self.assertEqual(evaluation_identity[0], prediction_identity[0])
        self.assertEqual(evaluation_identity[1], prediction_identity[1])

    def _create_test_artifact(
        self,
        temp_dir: str,
        report: EvaluationReport,
        replay_report: ReplayEvaluationReport | None = None,
        replay_suite_report: ReplaySuiteReport | None = None,
        created_at: str = "2026-05-31T120102Z",
        git_sha: str = "abc1234",
    ) -> object:
        model_path = Path(temp_dir) / f"model-{created_at}-{git_sha}.json"
        model_path.write_text("{}", encoding="utf-8")
        return create_model_artifact_bundle(
            model_path=model_path,
            backend="xgboost",
            training_config={
                "backend": "xgboost",
                "laps": 2,
                "seeds": 3,
                "rounds": 4,
                "training_rows": 18,
            },
            evaluation_report=report,
            replay_evaluation_report=replay_report or self._artifact_replay_report(),
            replay_suite_report=replay_suite_report or self._artifact_replay_suite_report(),
            artifact_root=Path(temp_dir) / "artifacts",
            created_at=created_at,
            git_sha=git_sha,
        )

    def _artifact_test_report(
        self,
        mae_lap_delta_s: float = 0.1,
        coverage_pct: float = 100.0,
        latency_p95_ms: float = 1.0,
        monotonic_wear_violations: int = 0,
    ) -> EvaluationReport:
        return EvaluationReport(
            version=APP_VERSION,
            feature_schema_version="online-features-v1",
            feature_schema_hash=feature_schema_hash(),
            scenarios=[
                ScenarioEvaluation(
                    scenario="unit",
                    laps=1,
                    compound="medium",
                    observations=1,
                    mae_lap_delta_s=mae_lap_delta_s,
                    rmse_lap_delta_s=mae_lap_delta_s,
                    mean_interval_width_s=0.4,
                    coverage_pct=coverage_pct,
                    latency_p95_ms=latency_p95_ms,
                    monotonic_wear_violations=monotonic_wear_violations,
                )
            ],
        )

    def _artifact_replay_report(
        self,
        passed: bool = True,
        mae_lap_delta_s: float = 0.1,
        coverage_pct: float = 100.0,
        latency_p95_ms: float = 1.0,
        missing_target_pct: float = 0.0,
        monotonic_wear_violations: int = 0,
    ) -> ReplayEvaluationReport:
        return ReplayEvaluationReport(
            version=APP_VERSION,
            feature_schema_version="online-features-v1",
            feature_schema_hash=feature_schema_hash(),
            dataset_path="examples/replay_telemetry.csv",
            dataset_fingerprint="replay-fingerprint",
            session_count=1,
            event_count=12,
            labeled_event_count=12,
            missing_target_pct=missing_target_pct,
            scenario=ScenarioEvaluation(
                scenario="replay-unit",
                laps=1,
                compound="mixed",
                observations=12,
                mae_lap_delta_s=mae_lap_delta_s,
                rmse_lap_delta_s=mae_lap_delta_s,
                mean_interval_width_s=0.4,
                coverage_pct=coverage_pct,
                latency_p95_ms=latency_p95_ms,
                monotonic_wear_violations=monotonic_wear_violations,
                source="replay",
                event_count=12,
            ),
            gates={
                "mae_lap_delta": passed,
                "coverage": passed,
                "calibration": passed,
                "sharpness": passed,
                "latency": passed,
                "monotonic_wear": passed,
                "target_completeness": passed,
                "sample_size": passed,
                "pit_decision": passed,
                "strategy_regret": passed,
            },
            passed=passed,
        )

    def _artifact_replay_suite_report(self, passed: bool = True) -> ReplaySuiteReport:
        split = self._artifact_replay_report(passed=passed)
        return ReplaySuiteReport(
            version=APP_VERSION,
            feature_schema_version="online-features-v1",
            feature_schema_hash=feature_schema_hash(),
            split_count=1,
            passed=passed,
            mean_mae_lap_delta_s=split.scenario.mae_lap_delta_s,
            mean_coverage_pct=split.scenario.coverage_pct,
            total_event_count=split.event_count,
            total_labeled_event_count=split.labeled_event_count,
            splits=[split],
        )

    def test_duckdb_persistence_lists_run_summaries_when_available(self) -> None:
        try:
            import duckdb  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"DuckDB is not installed: {exc}")

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.duckdb"
            persistence = DuckDBPersistence(path)
            engine = InferenceEngine(persistence=persistence)
            simulator = RaceSimulator(
                SimulationConfig(session_id="sim-race-history", laps=2, seed=14)
            )
            for event in simulator.events():
                prediction = engine.ingest(event)
            engine.strategy(prediction.session_id, prediction.car_id, remaining_laps=20)

            summaries = persistence.run_summaries()
            persistence.close()

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["session_id"], "sim-race-history")
        self.assertEqual(summaries[0]["latest_lap"], 2)
        self.assertEqual(summaries[0]["prediction_count"], 6)
        self.assertIsNotNone(summaries[0]["latest_strategy"])

    def test_live_simulation_manager_steps_events(self) -> None:
        manager = LiveSimulationManager()
        status = manager.start(laps=2, seed=12)
        self.assertTrue(status.running)
        batch = manager.tick(batch_size=2)
        self.assertEqual(len(batch), 2)
        self.assertEqual(manager.status()["index"], 2)

    def test_runtime_build_info_exposes_package_version(self) -> None:
        info = build_info()
        self.assertEqual(info.version, APP_VERSION)
        self.assertTrue(info.version)

    def test_xgboost_auto_backend_falls_back_without_artifact(self) -> None:
        model = create_serving_model(
            config=ModelConfig(),
            backend="auto",
            xgboost_model_path="missing/xgboost_lap_delta.json",
            lightgbm_model_path="missing/lightgbm_lap_delta.txt",
            catboost_model_path="missing/catboost_lap_delta.cbm",
            sequence_model_path="missing/sequence_lap_delta.pt",
        )
        self.assertIsInstance(model, HybridOnlineEnsembleModel)

    def test_explicit_artifact_backends_require_dependency_and_artifact(self) -> None:
        for backend in ("xgboost", "lightgbm", "catboost", "lstm", "tft"):
            with self.subTest(backend=backend):
                with self.assertRaises(RuntimeError):
                    create_serving_model(
                        config=ModelConfig(),
                        backend=backend,
                        xgboost_model_path="missing/xgboost_lap_delta.json",
                        lightgbm_model_path="missing/lightgbm_lap_delta.txt",
                        catboost_model_path="missing/catboost_lap_delta.cbm",
                        sequence_model_path="missing/sequence_lap_delta.pt",
                    )

    def test_kalman_backend_serves_without_optional_dependencies(self) -> None:
        model = create_serving_model(config=ModelConfig(), backend="kalman")
        self.assertIsInstance(model, KalmanFilterModel)
        engine = InferenceEngine(model=model)
        event = RaceSimulator(SimulationConfig(laps=1, seed=8)).events()[-1]
        prediction = engine.ingest(event)
        self.assertLess(prediction.uncertainty_low_s, prediction.uncertainty_high_s)

    def test_api_health_reports_runtime_version_when_available(self) -> None:
        try:
            from fastapi.testclient import TestClient

            from f1_strategy.api import app
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"FastAPI test dependencies are not installed: {exc}")

        self.assertIsNotNone(app)
        assert app is not None
        response = TestClient(app).get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["version"], APP_VERSION)
        self.assertEqual(payload["feature_schema_hash"], feature_schema_hash())
        self.assertIn("persistence_backend", payload)

    def test_api_rejects_invalid_telemetry_payload_when_available(self) -> None:
        try:
            from fastapi.testclient import TestClient

            from f1_strategy.api import app
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"FastAPI test dependencies are not installed: {exc}")

        self.assertIsNotNone(app)
        assert app is not None
        response = TestClient(app).post(
            "/telemetry",
            json={
                "session_id": "sim-race",
                "car_id": "car-16",
                "lap": 1,
                "sector": 4,
                "speed_kph": 240.0,
                "throttle": 1.2,
                "brake": 0.2,
                "steering_angle": 4.0,
                "tire_temp_fl": 95.0,
                "tire_temp_fr": 95.0,
                "tire_temp_rl": 92.0,
                "tire_temp_rr": 92.0,
                "brake_temp": 700.0,
                "slip_angle": 3.0,
                "lateral_g": 3.0,
                "ers_soc": 0.7,
                "ers_deployment_kw": 70.0,
                "fuel_kg": 60.0,
                "track_temp_c": 38.0,
                "air_temp_c": 27.0,
                "humidity": 0.45,
                "compound": "medium",
            },
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
