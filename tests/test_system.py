from __future__ import annotations

import csv
import json
import os
import unittest
from datetime import timedelta
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from f1_strategy.config import load_settings
from f1_strategy.shadow import ShadowDeploymentManager
from f1_strategy.data_sources.fastf1_export import (
    FastF1ExportConfig,
    export_manifest,
    manifest_path_for,
    rows_from_fastf1_session,
    write_replay_csv,
)
from f1_strategy.domain import OnlineFeatures, Prediction, TireCompound
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
    BENCHMARK_REPLAY_SUITE,
    REPLAY_REQUIRED_COLUMNS,
    ReplayEvaluationReport,
    ReplaySplitSpec,
    ReplaySuiteReport,
    benchmark_manifest_payload,
    load_replay_events,
    replay_data_provenance,
    run_benchmark_replay_suite,
    replay_report_to_dict,
    replay_suite_to_dict,
    run_replay_evaluation,
    run_replay_suite,
    write_benchmark_manifests,
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


class CountingModel:
    backend_name = "counting"

    def __init__(self) -> None:
        self.observed_features: list[OnlineFeatures] = []

    def observe(self, features: OnlineFeatures) -> None:
        self.observed_features.append(features)

    def predict(self, features: OnlineFeatures) -> Prediction:
        return Prediction(
            session_id=features.session_id,
            car_id=features.car_id,
            lap=features.lap,
            tire_wear_pct=0.0,
            remaining_tire_life_laps=30.0,
            grip_loss_pct=0.0,
            overheating_probability=0.0,
            cliff_probability=0.0,
            brake_temp_next_lap_c=500.0,
            ers_efficiency=features.ers_efficiency,
            next_lap_delta_s=0.0,
            uncertainty_low_s=-1.0,
            uncertainty_high_s=1.0,
        )


class FakeTelemetryFrame:
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def add_distance(self) -> "FakeTelemetryFrame":
        return self

    def to_dict(self, orientation: str) -> list[dict]:
        assert orientation == "records"
        return self.records


class FakeLap(dict):
    def __init__(self, payload: dict, telemetry: list[dict]) -> None:
        super().__init__(payload)
        self.telemetry = telemetry

    def get_car_data(self) -> FakeTelemetryFrame:
        return FakeTelemetryFrame(self.telemetry)


class FakeLaps(list):
    def pick_driver(self, driver: str) -> "FakeLaps":
        return FakeLaps(
            [
                lap
                for lap in self
                if lap.get("Driver") == driver or str(lap.get("DriverNumber")) == str(driver)
            ]
        )


class FakeWeatherFrame:
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def to_dict(self, orientation: str) -> list[dict]:
        assert orientation == "records"
        return self.records


class FakeFastF1Session:
    def __init__(self) -> None:
        self.laps = FakeLaps(
            [
                FakeLap(
                    {
                        "Driver": "VER",
                        "DriverNumber": 1,
                        "LapNumber": 1,
                        "LapTime": timedelta(seconds=91.2),
                        "Sector1Time": timedelta(seconds=30.1),
                        "Sector2Time": timedelta(seconds=31.0),
                        "Sector3Time": timedelta(seconds=30.1),
                        "Compound": "SOFT",
                        "Time": timedelta(seconds=91.2),
                    },
                    [
                        {
                            "Speed": 230 + index,
                            "Throttle": 70 + index,
                            "Brake": 5 if index < 4 else 30,
                            "Distance": index * 90.0,
                            "Time": timedelta(seconds=index * 2),
                            "X": index * 10.0,
                            "Y": index * 4.0,
                        }
                        for index in range(1, 10)
                    ],
                ),
                FakeLap(
                    {
                        "Driver": "VER",
                        "DriverNumber": 1,
                        "LapNumber": 2,
                        "LapTime": timedelta(seconds=91.5),
                        "Sector1Time": timedelta(seconds=30.2),
                        "Sector2Time": timedelta(seconds=31.2),
                        "Sector3Time": timedelta(seconds=30.1),
                        "Compound": "SOFT",
                        "Time": timedelta(seconds=182.7),
                    },
                    [
                        {
                            "Speed": 232 + index,
                            "Throttle": 72 + index,
                            "Brake": 10 if index < 5 else 35,
                            "Distance": index * 92.0,
                            "Time": timedelta(seconds=92 + index * 2),
                            "X": index * 11.0,
                            "Y": index * 3.0,
                        }
                        for index in range(1, 10)
                    ],
                ),
            ]
        )
        self.weather_data = FakeWeatherFrame(
            [
                {
                    "Time": timedelta(seconds=0),
                    "TrackTemp": 39.0,
                    "AirTemp": 28.0,
                    "Humidity": 48.0,
                },
                {
                    "Time": timedelta(seconds=120),
                    "TrackTemp": 40.0,
                    "AirTemp": 28.5,
                    "Humidity": 47.0,
                },
            ]
        )


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

    def test_replay_evaluation_accepts_openf1_tire_age_column(self) -> None:
        events = load_replay_events("examples/replay_telemetry.csv")
        sample = to_jsonable(events[0])
        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "openf1-replay.csv"
            fieldnames = list(REPLAY_REQUIRED_COLUMNS) + ["lap_time_s", "timestamp_ms", "circuit", "actual_tire_age_laps"]
            with dataset_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({**sample, "actual_tire_age_laps": 11})

            loaded = load_replay_events(dataset_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].session_id, sample["session_id"])

            report = run_replay_evaluation(dataset_path)
            self.assertEqual(report.event_count, 1)
            self.assertEqual(report.labeled_event_count, 1)

    def test_fastf1_export_maps_observed_and_derived_replay_fields(self) -> None:
        session = FakeFastF1Session()
        rows = rows_from_fastf1_session(
            session,
            year=2024,
            event="Bahrain",
            session_name="R",
            driver="VER",
        )

        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["session_id"], "fastf1-2024-bahrain-r")
        self.assertEqual(rows[0]["car_id"], "1")
        self.assertEqual(rows[0]["compound"], TireCompound.SOFT.value)
        self.assertEqual({row["sector"] for row in rows}, {1, 2, 3})
        self.assertTrue(all(row["lap_time_s"] for row in rows))
        self.assertTrue(all(row["speed_kph"] > 0 for row in rows))
        self.assertTrue(all(0.0 <= row["throttle"] <= 1.0 for row in rows))
        self.assertTrue(all(0.0 <= row["brake"] <= 1.0 for row in rows))
        self.assertTrue(all(row["tire_temp_fl"] > row["air_temp_c"] for row in rows))

    def test_fastf1_export_writes_csv_and_manifest_provenance(self) -> None:
        rows = rows_from_fastf1_session(
            FakeFastF1Session(),
            year=2024,
            event="Bahrain",
            session_name="R",
            driver="VER",
            max_laps=1,
        )
        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "fastf1_replay.csv"
            write_replay_csv(output, rows)
            self.assertTrue(output.exists())
            self.assertEqual(manifest_path_for(output).name, "fastf1_replay.csv.manifest.json")

            manifest = export_manifest(
                config=FastF1ExportConfig(
                    year=2024,
                    event="Bahrain",
                    session="R",
                    driver="VER",
                    output=output,
                ),
                rows=rows,
                fastf1_version="test",
            )

        self.assertEqual(manifest["source"], "fastf1")
        self.assertEqual(manifest["row_count"], 3)
        self.assertEqual(manifest["field_provenance"]["lap_time_s"].split(":")[0], "observed")

    def test_fastf1_manifest_marks_observed_public_validation_signal(self) -> None:
        rows = rows_from_fastf1_session(
            FakeFastF1Session(),
            year=2024,
            event="Bahrain",
            session_name="R",
            driver="VER",
            max_laps=1,
        )
        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "fastf1_replay.csv"
            write_replay_csv(output, rows)
            manifest = export_manifest(
                config=FastF1ExportConfig(
                    year=2024,
                    event="Bahrain",
                    session="R",
                    driver="VER",
                    output=output,
                ),
                rows=rows,
                fastf1_version="test",
            )
            manifest_path_for(output).write_text(json.dumps(manifest), encoding="utf-8")

            provenance = replay_data_provenance(output)

        self.assertEqual(provenance["validation_signal"], "observed-public")
        self.assertTrue(provenance["production_validation_ready"])
        self.assertIn("lap_time_s", provenance["observed_fields"])
        self.assertIn("derived", manifest["field_provenance"]["tire_temp_fl"])

    def test_replay_suite_reports_named_splits(self) -> None:
        report = run_replay_suite()
        payload = replay_suite_to_dict(report)

        self.assertGreaterEqual(report.split_count, 5)
        self.assertTrue(report.passed, payload)
        self.assertIn("smoke", {split.scenario.scenario for split in report.splits})
        self.assertGreater(report.total_event_count, 12)
        self.assertTrue(all(split.dataset_fingerprint for split in report.splits))

    def test_benchmark_replay_suite_enforces_committed_slices(self) -> None:
        report = run_benchmark_replay_suite()
        payload = replay_suite_to_dict(report)

        self.assertEqual(report.suite_name, "benchmark")
        self.assertEqual(report.split_count, len(BENCHMARK_REPLAY_SUITE))
        self.assertGreaterEqual(report.total_event_count, 200)
        self.assertEqual(report.total_event_count, report.total_labeled_event_count)
        self.assertTrue(report.passed, payload)

    def test_benchmark_replay_fixtures_have_non_production_manifests(self) -> None:
        for spec in BENCHMARK_REPLAY_SUITE:
            assert spec.dataset_path is not None
            provenance = replay_data_provenance(spec.dataset_path)

            self.assertTrue(provenance["manifest_available"], spec.dataset_path)
            self.assertEqual(provenance["source"], "benchmark-fixture")
            self.assertEqual(provenance["validation_signal"], "proxy-heavy")
            self.assertEqual(provenance["lap_time_label"], "synthetic")
            self.assertFalse(provenance["production_validation_ready"])

    def test_benchmark_manifest_generator_matches_committed_sidecars(self) -> None:
        checked = write_benchmark_manifests(check=True)

        self.assertEqual(len(checked), len(BENCHMARK_REPLAY_SUITE))
        for spec in BENCHMARK_REPLAY_SUITE:
            assert spec.dataset_path is not None
            manifest_path = spec.dataset_path.with_suffix(spec.dataset_path.suffix + ".manifest.json")
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(actual, benchmark_manifest_payload(spec))

    def test_ui_static_replay_trust_badges_are_wired(self) -> None:
        app_js = Path("src/f1_strategy/ui/app.js").read_text(encoding="utf-8")
        index_html = Path("src/f1_strategy/ui/index.html").read_text(encoding="utf-8")
        styles = Path("src/f1_strategy/ui/styles.css").read_text(encoding="utf-8")

        for label in ["Production", "Benchmark", "Synthetic", "Proxy", "No Manifest"]:
            self.assertIn(label, app_js)
        self.assertIn("Label Type", index_html)
        self.assertIn("trust-badge", styles)
        self.assertIn("trustBadge(provenance)", app_js)

    def test_ui_static_external_links_are_api_driven(self) -> None:
        app_js = Path("src/f1_strategy/ui/app.js").read_text(encoding="utf-8")
        index_html = Path("src/f1_strategy/ui/index.html").read_text(encoding="utf-8")
        styles = Path("src/f1_strategy/ui/styles.css").read_text(encoding="utf-8")

        self.assertIn("/integrations/external-links", app_js)
        self.assertNotIn("/integrations/external-links?probe=1", app_js)
        self.assertIn("resolveInfraUrl", app_js)
        self.assertIn("defaultExternalServices", app_js)
        self.assertIn("mergeExternalServices", app_js)
        self.assertIn("runReplayCheck", app_js)
        self.assertIn("runRegressionCheck", app_js)
        self.assertIn("runSmokeCheck", app_js)
        self.assertNotIn("window.location.hostname", app_js)
        self.assertIn("externalLinksRow", index_html)
        self.assertIn("Evaluation", index_html)
        self.assertIn("Promotion", index_html)
        self.assertIn("Training", index_html)
        self.assertIn("Run Checks", index_html)
        self.assertIn("Replay Run", index_html)
        self.assertIn("Regression Run", index_html)
        self.assertIn("Run Deployment Smoke", index_html)
        self.assertIn("trainingReplayDatasetSelect", index_html)
        self.assertIn("Auto (serving)", index_html)
        self.assertIn("comparisonSummary", index_html)
        self.assertIn("comparisonModelValue", index_html)
        self.assertIn("comparisonVerdictValue", index_html)
        self.assertIn("comparisonActionValue", index_html)
        self.assertIn("Action", index_html)
        self.assertNotIn('href="http://localhost:5000"', index_html)
        self.assertNotIn('href="http://localhost:3000"', index_html)
        self.assertNotIn('href="http://localhost:9090"', index_html)
        self.assertIn("replay_dataset_path", app_js)
        self.assertIn(".ext-link.disabled", styles)

    def test_ui_tab_sections_are_balanced(self) -> None:
        index_html = Path("src/f1_strategy/ui/index.html").read_text(encoding="utf-8")

        class SectionBalanceParser(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.unclosed: list[tuple[int, int]] = []

            def handle_starttag(self, tag: str, attrs) -> None:
                if tag == "section":
                    self.unclosed.append(self.getpos())

            def handle_endtag(self, tag: str) -> None:
                if tag == "section" and self.unclosed:
                    self.unclosed.pop()

        parser = SectionBalanceParser()
        parser.feed(index_html)
        self.assertFalse(parser.unclosed, f"Unclosed section tags at: {parser.unclosed}")

    def test_replay_suite_default_engine_ignores_serving_artifact_environment(self) -> None:
        created_settings = []
        real_engine = InferenceEngine

        def engine_factory(*args, **kwargs):
            settings = kwargs.get("settings")
            if settings is not None:
                created_settings.append(settings)
            return real_engine(*args, **kwargs)

        with patch.dict(
            os.environ,
            {
                "F1_MODEL_BACKEND": "kalman",
                "F1_MODEL_ARTIFACT_ID": "xgboost/does-not-exist",
            },
            clear=False,
        ), patch("f1_strategy.replay.InferenceEngine", side_effect=engine_factory):
            report = run_replay_suite([ReplaySplitSpec("backend-default", laps=4, seed=301)])

        self.assertTrue(report.passed, replay_suite_to_dict(report))
        self.assertTrue(created_settings)
        self.assertTrue(all(settings.model_backend == "hybrid" for settings in created_settings))
        self.assertTrue(all(settings.model_artifact_id == "" for settings in created_settings))

    def test_online_model_observes_only_labeled_lap_events(self) -> None:
        model = CountingModel()
        engine = InferenceEngine(model=model)
        events = RaceSimulator(SimulationConfig(laps=2, seed=31)).events()
        unlabeled = [event for event in events if event.lap_time_s is None][:2]
        labeled = next(event for event in events if event.lap_time_s is not None)

        for event in unlabeled:
            engine.ingest(event)
        self.assertEqual(model.observed_features, [])

        engine.ingest(labeled)
        self.assertEqual(len(model.observed_features), 1)
        self.assertEqual(model.observed_features[0].lap, labeled.lap)

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
            self.assertTrue((bundle.bundle_dir / "model_card.json").exists())
            manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
            model_card = json.loads((bundle.bundle_dir / "model_card.json").read_text())
            self.assertEqual(manifest["replay_dataset_fingerprint"], "replay-fingerprint")
            self.assertEqual(manifest["model_card"], "model_card.json")
            self.assertEqual(model_card["artifact_id"], bundle.artifact_id)
            self.assertIn("replay_evaluation", model_card)
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

    def test_promote_artifact_requires_benchmark_replay_suite(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bundle = self._create_test_artifact(
                temp_dir,
                report=self._artifact_test_report(),
                replay_suite_report=self._artifact_replay_suite_report(suite_name="smoke"),
            )

            result = promote_artifact(
                bundle.artifact_id,
                artifact_root=Path(temp_dir) / "artifacts",
            )

            self.assertFalse(result.promoted)
            self.assertTrue(any("suite name" in failure for failure in result.failures))

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
        with TemporaryDirectory() as temp_dir:
            settings = load_settings().__class__(
                model_artifact_root=str(Path(temp_dir) / "artifacts")
            )
            engine = InferenceEngine(settings=settings)
            readiness = engine.deployment_readiness()
            metrics = engine.monitoring.render_prometheus()

            self.assertIn("ready", readiness)
            self.assertIn("checks", readiness)
            self.assertIn("rollback_candidate", readiness)
            self.assertIn("f1_deployment_ready", metrics)

            production = engine.deployment_readiness(mode="production")
            self.assertFalse(production["ready"])
            self.assertEqual(production["mode"], "production")
            self.assertIn("production_replay_validation", production["checks"])

    def test_strict_promotion_rejects_artifact_without_production_replay_signal(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bundle = self._create_test_artifact(temp_dir, report=self._artifact_test_report())

            result = promote_artifact(
                bundle.artifact_id,
                artifact_root=Path(temp_dir) / "artifacts",
                gates=PromotionGateConfig(require_production_replay_validation=True),
            )

        self.assertFalse(result.promoted)
        self.assertTrue(
            any("production validation" in failure for failure in result.failures),
            result.failures,
        )

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
            feature_schema_version="online-features-v2",
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
            feature_schema_version="online-features-v2",
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

    def _artifact_replay_suite_report(
        self,
        passed: bool = True,
        suite_name: str = "benchmark",
    ) -> ReplaySuiteReport:
        split = self._artifact_replay_report(passed=passed)
        splits = [split] * 5
        return ReplaySuiteReport(
            version=APP_VERSION,
            feature_schema_version="online-features-v2",
            feature_schema_hash=feature_schema_hash(),
            split_count=len(splits),
            passed=passed,
            mean_mae_lap_delta_s=split.scenario.mae_lap_delta_s,
            mean_coverage_pct=split.scenario.coverage_pct,
            total_event_count=sum(item.event_count for item in splits),
            total_labeled_event_count=sum(item.labeled_event_count for item in splits),
            splits=splits,
            suite_name=suite_name,
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


class AutoDriftTests(unittest.TestCase):
    """Drift detection fires automatically inside engine.ingest()."""

    def _run_engine(self, laps: int = 12) -> InferenceEngine:
        from f1_strategy.simulation import RaceSimulator, SimulationConfig
        engine = InferenceEngine()
        sim = RaceSimulator(SimulationConfig(
            session_id="drift-test", car_id="c1", laps=laps, seed=7
        ))
        for event in sim.events():
            engine.ingest(event)
        return engine

    def test_baseline_auto_fitted_after_warmup(self) -> None:
        engine = self._run_engine(laps=15)
        # Baseline is fitted after DRIFT_WARMUP (30) events; 15 laps × 3 sectors = 45
        self.assertGreater(len(engine.drift_detector._baseline), 0)

    def test_drift_scores_appear_in_prometheus_after_warmup(self) -> None:
        engine = self._run_engine(laps=15)
        metrics = engine.monitoring.render_prometheus()
        drift_lines = [line for line in metrics.splitlines() if "f1_drift_z_score_" in line and not line.startswith("#")]
        self.assertGreater(len(drift_lines), 0)

    def test_psi_scores_emitted_after_window_fills(self) -> None:
        from f1_strategy.drift import DriftDetector
        from f1_strategy.simulation import RaceSimulator, SimulationConfig
        from f1_strategy.feature_store import OnlineFeatureStore

        d = DriftDetector(window_size=10)
        sim = RaceSimulator(SimulationConfig(session_id="s", car_id="c", laps=20, seed=1))
        store = OnlineFeatureStore()
        features_list = [store.ingest(ev) for ev in sim.events()]

        d.fit_baseline(features_list[:15])
        report = None
        for f in features_list[15:]:
            report = d.detect(f)

        self.assertIsNotNone(report)
        psi_keys = [k for k in report.feature_scores if k.startswith("psi_")]
        self.assertGreater(len(psi_keys), 0, "expected PSI scores after window fills")

    def test_concept_drift_z_tracked_after_error_warmup(self) -> None:
        from f1_strategy.drift import DriftDetector
        from f1_strategy.simulation import RaceSimulator, SimulationConfig
        from f1_strategy.feature_store import OnlineFeatureStore

        d = DriftDetector(window_size=15)
        sim = RaceSimulator(SimulationConfig(session_id="s", car_id="c", laps=25, seed=2))
        store = OnlineFeatureStore()
        features_list = [store.ingest(ev) for ev in sim.events()]

        d.fit_baseline(features_list[:20])
        for i, f in enumerate(features_list[20:]):
            d.record_error(0.05 * ((i % 8) + 1))
            report = d.detect(f)

        self.assertIn("concept_drift_z", report.feature_scores)


class ShadowDeploymentTests(unittest.TestCase):
    """ShadowDeploymentManager records divergence and surfaces promotion candidates."""

    def _make_shadow_with_data(self, n_events: int = 60) -> tuple:
        from f1_strategy.models import KalmanFilterModel, ModelConfig
        from f1_strategy.simulation import RaceSimulator, SimulationConfig

        engine = InferenceEngine()
        challenger = KalmanFilterModel(ModelConfig())
        engine.shadow.configure(model=challenger, backend="kalman")

        sim = RaceSimulator(SimulationConfig(session_id="s1", car_id="c1", laps=25, seed=3))
        for event in sim.events():
            engine.ingest(event)
            if engine.shadow._total >= n_events:
                break
        return engine, engine.shadow

    def test_shadow_records_divergence_between_models(self) -> None:
        engine, shadow = self._make_shadow_with_data(n_events=10)
        self.assertTrue(shadow.active)
        self.assertGreater(shadow._total, 0)
        status = shadow.status()
        self.assertIn("divergence_rate", status)
        self.assertIn("mean_abs_delta_s", status)
        self.assertGreaterEqual(status["mean_abs_delta_s"], 0.0)

    def test_shadow_promotion_candidate_none_with_insufficient_data(self) -> None:
        from f1_strategy.models import KalmanFilterModel, ModelConfig
        shadow = ShadowDeploymentManager()
        shadow.configure(model=KalmanFilterModel(ModelConfig()), backend="kalman")
        # No data recorded yet
        self.assertIsNone(shadow.promotion_candidate(min_predictions=50))

    def test_shadow_promotion_candidate_returned_when_challenger_better(self) -> None:
        shadow = ShadowDeploymentManager(window_size=200)
        from f1_strategy.models import KalmanFilterModel, ModelConfig
        shadow.configure(model=KalmanFilterModel(ModelConfig()), backend="kalman")

        # Inject synthetic window where challenger has lower absolute delta
        for i in range(60):
            shadow._window.append({
                "lap": i + 1,
                "champion_delta_s": 0.50,
                "challenger_delta_s": 0.20,
                "delta_s": 0.30,
                "abs_delta_s": 0.30,
                "champion_wear": 30.0,
                "challenger_wear": 30.0,
                "champion_cliff": 0.1,
                "challenger_cliff": 0.1,
                "challenger_latency_ms": 1.0,
            })
        shadow._total = 60  # must match window size for promotion to trigger

        candidate = shadow.promotion_candidate(min_predictions=50, improvement_threshold_s=0.02)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["challenger_backend"], "kalman")
        self.assertGreater(candidate["improvement_s"], 0)
        self.assertIn("recommendation", candidate)

    def test_shadow_disable_clears_all_state(self) -> None:
        from f1_strategy.models import KalmanFilterModel, ModelConfig
        shadow = ShadowDeploymentManager()
        shadow.configure(model=KalmanFilterModel(ModelConfig()), backend="kalman")
        shadow._total = 5
        shadow.disable()
        self.assertFalse(shadow.active)
        self.assertEqual(shadow._total, 0)
        self.assertEqual(len(shadow._window), 0)
        self.assertIsNone(shadow.promotion_candidate())

    def test_status_includes_promotion_candidate_key(self) -> None:
        shadow = ShadowDeploymentManager()
        status = shadow.status()
        self.assertIn("promotion_candidate", status)


class LiveTimingTests(unittest.TestCase):
    """ReplayStreamSource streams CSV events at configurable speed."""

    def test_replay_stream_emits_all_labeled_rows(self) -> None:
        import threading
        import time
        from f1_strategy.data_sources.live_timing import ReplayStreamSource

        received = []
        lock = threading.Lock()

        def on_event(ev):
            with lock:
                received.append(ev)

        src = ReplayStreamSource(
            dataset_path="examples/replay_telemetry.csv",
            speed_multiplier=500.0,
            on_event=on_event,
        )
        src.start()
        time.sleep(1.5)
        src.stop()

        with lock:
            count = len(received)
        self.assertGreater(count, 0)
        self.assertIsNotNone(received[0].session_id)

    def test_replay_stream_status_reflects_progress(self) -> None:
        import time
        from f1_strategy.data_sources.live_timing import ReplayStreamSource

        src = ReplayStreamSource(
            dataset_path="examples/replay_telemetry.csv",
            speed_multiplier=500.0,
        )
        src.start()
        time.sleep(1.5)
        self.assertGreaterEqual(src.status.events_ingested, 0)

    def test_live_stream_manager_configure_sets_mode(self) -> None:
        from f1_strategy.data_sources.live_timing import LiveStreamManager
        mgr = LiveStreamManager()
        status = mgr.configure_replay("examples/replay_telemetry.csv", speed_multiplier=1.0)
        self.assertEqual(status.mode, "replay")

    def test_replay_stop_halts_stream(self) -> None:
        import time
        from f1_strategy.data_sources.live_timing import ReplayStreamSource

        received = []

        def on_event(ev):
            received.append(ev)

        src = ReplayStreamSource(
            dataset_path="examples/replay_telemetry.csv",
            speed_multiplier=2.0,
            on_event=on_event,
        )
        src.start()
        time.sleep(0.1)
        src.stop()
        time.sleep(0.2)
        count_after_stop = len(received)
        time.sleep(0.3)
        # no new events should arrive after stop
        self.assertEqual(len(received), count_after_stop)


class RealDataRowsTests(unittest.TestCase):
    """_real_data_rows() extracts labeled feature vectors from a replay CSV."""

    def _write_temp_csv(self, tmp_dir: str, include_labeled: bool = True) -> str:
        import csv as _csv
        from f1_strategy.data_sources.fastf1_export import REPLAY_COLUMNS

        path = f"{tmp_dir}/test_real.csv"
        rows = [
            {
                "session_id": "test", "car_id": "VER", "lap": 1, "sector": 1,
                "speed_kph": 230.0, "throttle": 0.85, "brake": 0.1,
                "steering_angle": -5.0, "tire_temp_fl": 95.0, "tire_temp_fr": 95.0,
                "tire_temp_rl": 92.0, "tire_temp_rr": 92.0, "brake_temp": 600.0,
                "slip_angle": 2.5, "lateral_g": 2.8, "ers_soc": 0.75,
                "ers_deployment_kw": 70.0, "fuel_kg": 95.0, "track_temp_c": 38.0,
                "air_temp_c": 27.0, "humidity": 0.44, "compound": "medium",
                "lap_time_s": "91.234" if include_labeled else "",
                "timestamp_ms": 10000,
            }
            for _ in range(5)
        ]
        with open(path, "w", newline="") as fh:
            writer = _csv.DictWriter(fh, fieldnames=REPLAY_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({col: row.get(col, "") for col in REPLAY_COLUMNS})
        return path

    def test_real_data_rows_reads_labeled_csv(self) -> None:
        import tempfile
        from f1_strategy.training import _real_data_rows
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_temp_csv(tmp, include_labeled=True)
            rows, targets = _real_data_rows(path)
        self.assertGreater(len(rows), 0)
        self.assertEqual(len(rows), len(targets))
        self.assertTrue(all(isinstance(r, list) for r in rows))

    def test_real_data_rows_skips_unlabeled(self) -> None:
        import tempfile
        from f1_strategy.training import _real_data_rows
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_temp_csv(tmp, include_labeled=False)
            rows, targets = _real_data_rows(path)
        self.assertEqual(len(rows), 0)


class RedisFeatureStoreTests(unittest.TestCase):
    """RedisFeatureStore degrades to in-memory when Redis is unavailable."""

    def test_create_feature_store_falls_back_to_in_memory_without_redis(self) -> None:
        from f1_strategy.feature_store import OnlineFeatureStore, create_feature_store
        store = create_feature_store(backend="auto", redis_url="redis://127.0.0.1:19999/0")
        self.assertIsInstance(store, OnlineFeatureStore)

    def test_create_feature_store_returns_in_memory_when_no_url(self) -> None:
        from f1_strategy.feature_store import OnlineFeatureStore, create_feature_store
        store = create_feature_store(backend="auto", redis_url="")
        self.assertIsInstance(store, OnlineFeatureStore)

    def test_redis_feature_store_ingest_works_when_redis_unavailable(self) -> None:
        from f1_strategy.feature_store import RedisFeatureStore
        from f1_strategy.simulation import RaceSimulator, SimulationConfig
        store = RedisFeatureStore(redis_url="redis://127.0.0.1:19999/0", window_size=20)
        self.assertFalse(store.is_connected)  # no Redis at that port
        sim = RaceSimulator(SimulationConfig(session_id="s", car_id="c", laps=3, seed=1))
        events = list(sim.events())
        features = store.ingest(events[0])
        self.assertEqual(features.session_id, "s")


if __name__ == "__main__":
    unittest.main()
