from __future__ import annotations

import unittest


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import f1_strategy.api as api_module
        except ImportError as exc:
            raise unittest.SkipTest(f"API dependencies are not installed: {exc}") from exc
        if api_module.app is None:
            raise unittest.SkipTest("FastAPI application is unavailable")
        cls.api = api_module

    def setUp(self) -> None:
        self.api.simulation_reset()

    def test_core_routes_are_registered(self) -> None:
        paths = {route.path for route in self.api.app.routes}
        self.assertIn("/", paths)
        self.assertIn("/health", paths)
        self.assertIn("/models", paths)
        self.assertIn("/artifacts", paths)
        self.assertIn("/model/backend", paths)
        self.assertIn("/model/artifact", paths)
        self.assertIn("/monitoring/model-performance", paths)
        self.assertIn("/monitoring/model-comparison", paths)
        self.assertIn("/monitoring/alerts", paths)
        self.assertIn("/deployment/readiness", paths)
        self.assertIn("/deployment/rollback-candidate", paths)
        self.assertIn("/simulation/start", paths)
        self.assertIn("/simulation/tick", paths)
        self.assertIn("/simulation/reset", paths)
        self.assertIn("/history/runs", paths)
        self.assertIn("/evaluation/replay", paths)
        self.assertIn("/evaluation/replay-suite", paths)
        self.assertIn("/artifacts/{artifact_id:path}", paths)

    def test_health_and_ui_index_are_available(self) -> None:
        health = self.api.health()
        self.assertEqual(health["status"], "ok")
        self.assertIn("persistence_backend", health)
        self.assertIn("model_artifact_id", health)
        self.assertGreater(health["target_latency_ms"], 0)

        response = self.api.index()
        self.assertTrue(str(response.path).endswith("index.html"))

    def test_model_backend_can_switch_to_dependency_light_model(self) -> None:
        models = self.api.models()
        self.assertIn("kalman", models["available_backends"])

        switched = self.api.model_backend("kalman")
        self.assertEqual(switched["configured_backend"], "kalman")
        self.assertEqual(switched["active_backend"], "kalman")
        self.assertEqual(switched["active_artifact_id"], "unregistered")
        self.assertEqual(self.api.health()["model_backend"], "kalman")

    def test_artifacts_route_lists_registry_shape(self) -> None:
        payload = self.api.artifacts()
        self.assertIn("artifact_root", payload)
        self.assertIn("active_artifact_id", payload)
        self.assertIn("promoted", payload)
        self.assertIn("artifacts", payload)
        self.assertIsInstance(payload["artifacts"], list)

    def test_simulation_start_tick_reset_and_history(self) -> None:
        status = self.api.simulation_start(laps=2, seed=42)
        self.assertTrue(status["running"])
        self.assertEqual(status["total"], 6)
        self.assertTrue(str(status["session_id"]).startswith("sim-race-"))

        tick = self.api.simulation_tick(batch_size=3)
        self.assertEqual(len(tick["telemetry"]), 3)
        self.assertEqual(len(tick["predictions"]), 3)
        self.assertIn("tire_temp_fl", tick["telemetry"][-1])
        self.assertEqual(tick["status"]["index"], 3)
        self.assertIn("metrics", tick)

        history = self.api.history_runs(limit=3)
        self.assertIn(history["persistence_backend"], {"memory", "duckdb"})
        self.assertIsInstance(history["runs"], list)

        performance = self.api.model_performance()
        self.assertIn("models", performance)
        self.assertIsInstance(performance["models"], list)
        comparison = self.api.model_comparison()
        self.assertIn("models", comparison)
        self.assertIsInstance(comparison["models"], list)
        metrics = self.api.metrics().body.decode("utf-8")
        self.assertIn("f1_model_mae_lap_delta_s", metrics)
        self.assertIn("f1_model_rmse_lap_delta_s", metrics)
        self.assertIn("f1_model_interval_coverage_pct", metrics)

        alerts = self.api.monitoring_alerts()
        self.assertIn("health_score", alerts)
        self.assertIn("alerts", alerts)

        readiness = self.api.deployment_readiness()
        self.assertIn("ready", readiness)
        self.assertIn("checks", readiness)

        replay = self.api.replay_evaluation()
        self.assertTrue(replay["passed"])
        self.assertEqual(replay["scenario"]["source"], "replay")
        self.assertIn("dataset_fingerprint", replay)
        replay_suite = self.api.replay_suite()
        self.assertTrue(replay_suite["passed"])
        self.assertGreaterEqual(replay_suite["split_count"], 5)

        reset = self.api.simulation_reset()
        self.assertFalse(reset["running"])
        self.assertEqual(reset["total"], 0)

    def test_telemetry_request_validation_rejects_invalid_payload(self) -> None:
        with self.assertRaises(Exception):
            self.api.TelemetryEventRequest(
                session_id="sim-test",
                car_id="car-16",
                lap=1,
                sector=4,
                speed_kph=240.0,
                throttle=1.4,
                brake=0.2,
                steering_angle=3.0,
                tire_temp_fl=94.0,
                tire_temp_fr=95.0,
                tire_temp_rl=91.0,
                tire_temp_rr=92.0,
                brake_temp=680.0,
                slip_angle=2.5,
                lateral_g=2.8,
                ers_soc=0.7,
                ers_deployment_kw=80.0,
                fuel_kg=55.0,
                track_temp_c=38.0,
                air_temp_c=27.0,
                humidity=0.5,
                compound="medium",
            )


if __name__ == "__main__":
    unittest.main()
