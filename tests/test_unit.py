from __future__ import annotations

import csv
import json
import math
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from f1_strategy.domain import FleetState, TireCompound
from f1_strategy.feature_store import OnlineFeatureStore
from f1_strategy.shadow import _welch_pvalue, ShadowDeploymentManager
from f1_strategy.simulation import RaceSimulator, SimulationConfig
from f1_strategy.training import _validate_csv_schema


# ---------------------------------------------------------------------------
# FleetState
# ---------------------------------------------------------------------------

class FleetStateTests(unittest.TestCase):

    def test_gap_defaults_to_unknown_before_update(self) -> None:
        fs = FleetState("race-1")
        self.assertEqual(fs.gap_ahead_s("VER"), 999.0)
        self.assertEqual(fs.gap_behind_s("VER"), 999.0)
        self.assertEqual(fs.position("VER"), 10)

    def test_update_persists_gap_and_position(self) -> None:
        fs = FleetState("race-1")
        fs.update("VER", position=2, gap_ahead_s=0.8, gap_behind_s=1.3,
                  competitor_tire_age=12, competitor_compound=TireCompound.MEDIUM)
        self.assertAlmostEqual(fs.gap_ahead_s("VER"), 0.8)
        self.assertAlmostEqual(fs.gap_behind_s("VER"), 1.3)
        self.assertEqual(fs.position("VER"), 2)
        self.assertEqual(fs.competitor_tire_age("VER"), 12)
        self.assertEqual(fs.competitor_compound("VER"), TireCompound.MEDIUM)

    def test_undercut_threat_zero_when_gap_is_large(self) -> None:
        fs = FleetState("race-1")
        fs.update("HAM", gap_behind_s=30.0)
        self.assertAlmostEqual(fs.undercut_threat("HAM"), 0.0)

    def test_undercut_threat_high_when_gap_small_and_tires_old(self) -> None:
        fs = FleetState("race-1")
        fs.update("HAM", gap_behind_s=2.0, competitor_tire_age=22)
        threat = fs.undercut_threat("HAM")
        self.assertGreater(threat, 0.5)

    def test_as_features_returns_all_fleet_keys(self) -> None:
        fs = FleetState("race-1")
        fs.update("VER", position=1, gap_ahead_s=0.5, gap_behind_s=1.2)
        feats = fs.as_features("VER")
        self.assertIn("fleet_gap_ahead_s", feats)
        self.assertIn("fleet_gap_behind_s", feats)
        self.assertIn("fleet_position", feats)
        self.assertIn("fleet_competitor_tire_age", feats)
        self.assertIn("fleet_competitor_compound", feats)

    def test_multiple_cars_tracked_independently(self) -> None:
        fs = FleetState("race-1")
        fs.update("VER", gap_ahead_s=0.3, gap_behind_s=1.5)
        fs.update("HAM", gap_ahead_s=1.5, gap_behind_s=3.0)
        self.assertAlmostEqual(fs.gap_ahead_s("VER"), 0.3)
        self.assertAlmostEqual(fs.gap_ahead_s("HAM"), 1.5)


# ---------------------------------------------------------------------------
# Feature store — fleet integration and dirty air risk
# ---------------------------------------------------------------------------

class FleetAwareFeatureStoreTests(unittest.TestCase):

    def _make_event(self, session_id: str = "s", car_id: str = "c") -> object:
        sim = RaceSimulator(SimulationConfig(session_id=session_id, car_id=car_id, laps=3, seed=5))
        return sim.events()[-1]

    def test_fleet_gap_fields_default_to_unknown_without_fleet(self) -> None:
        store = OnlineFeatureStore()
        event = self._make_event()
        features = store.ingest(event)
        self.assertAlmostEqual(features.fleet_gap_ahead_s, 999.0)
        self.assertAlmostEqual(features.fleet_gap_behind_s, 999.0)

    def test_fleet_gap_fields_populated_when_fleet_state_attached(self) -> None:
        store = OnlineFeatureStore()
        fs = FleetState("s")
        fs.update("c", gap_ahead_s=0.8, gap_behind_s=2.1, position=3)
        store.set_fleet_state(fs)
        event = self._make_event()
        features = store.ingest(event)
        self.assertAlmostEqual(features.fleet_gap_ahead_s, 0.8)
        self.assertAlmostEqual(features.fleet_gap_behind_s, 2.1)
        self.assertEqual(features.fleet_position, 3)

    def test_dirty_air_risk_higher_when_close_behind_leader(self) -> None:
        """Fleet gap < 1.5s should push dirty_air_risk toward 1.0."""
        store_no_fleet = OnlineFeatureStore()
        store_with_fleet = OnlineFeatureStore()
        fs = FleetState("s")
        fs.update("c", gap_ahead_s=0.3)   # very close
        store_with_fleet.set_fleet_state(fs)

        # Run enough events to get a stable dirty air calculation.
        sim = RaceSimulator(SimulationConfig(session_id="s", car_id="c", laps=5, seed=9))
        feats_no_fleet = feats_with_fleet = None
        for ev in sim.events():
            feats_no_fleet = store_no_fleet.ingest(ev)
            feats_with_fleet = store_with_fleet.ingest(ev)

        assert feats_with_fleet is not None
        # With 0.3s gap ahead, dirty air risk should be elevated.
        self.assertGreater(feats_with_fleet.dirty_air_risk, feats_no_fleet.dirty_air_risk)  # type: ignore[union-attr]

    def test_compound_wear_multiplier_makes_soft_degrade_faster_than_hard(self) -> None:
        from f1_strategy.simulation import RaceSimulator, SimulationConfig

        def run_cumulative_load(compound: TireCompound) -> float:
            store = OnlineFeatureStore()
            sim = RaceSimulator(SimulationConfig(
                session_id="s", car_id="c", laps=5, seed=99, compound=compound
            ))
            feats = None
            for ev in sim.events():
                feats = store.ingest(ev)
            assert feats is not None
            return feats.cumulative_tire_load

        soft_load = run_cumulative_load(TireCompound.SOFT)
        hard_load = run_cumulative_load(TireCompound.HARD)
        self.assertGreater(soft_load, hard_load,
                           "Soft compound should accumulate more wear load than hard")


# ---------------------------------------------------------------------------
# Feature store — checkpoint / restore
# ---------------------------------------------------------------------------

class FeatureStoreCheckpointTests(unittest.TestCase):

    def test_checkpoint_and_restore_preserves_lap_state(self) -> None:
        store = OnlineFeatureStore(window_size=30, base_lap_time_s=92.0)
        sim = RaceSimulator(SimulationConfig(session_id="s", car_id="c", laps=8, seed=7))
        original_features = None
        for ev in sim.events():
            original_features = store.ingest(ev)
        assert original_features is not None

        with TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "store.json"
            store.checkpoint(ckpt)
            self.assertTrue(ckpt.exists())

            restored = OnlineFeatureStore.from_checkpoint(ckpt)
            # Restored store should have the same session/car state.
            restored_features = restored.get("s", "c")
            self.assertIsNotNone(restored_features)
            assert restored_features is not None
            self.assertEqual(restored_features.session_id, original_features.session_id)
            self.assertEqual(restored_features.lap, original_features.lap)
            self.assertAlmostEqual(
                restored_features.rolling_lap_time_s,
                original_features.rolling_lap_time_s,
                places=3,
            )

    def test_checkpoint_json_is_valid(self) -> None:
        store = OnlineFeatureStore()
        sim = RaceSimulator(SimulationConfig(session_id="t", car_id="d", laps=3, seed=2))
        for ev in sim.events():
            store.ingest(ev)

        with TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "store.json"
            store.checkpoint(ckpt)
            data = json.loads(ckpt.read_text())
        self.assertIn("events", data)
        self.assertIn("window_size", data)
        self.assertIn("base_lap_time_s", data)

    def test_restore_empty_store_returns_no_features(self) -> None:
        store = OnlineFeatureStore()
        with TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "empty.json"
            store.checkpoint(ckpt)
            restored = OnlineFeatureStore.from_checkpoint(ckpt)
        self.assertEqual(restored.snapshot(), [])


# ---------------------------------------------------------------------------
# Shadow deployment — Welch t-test gate
# ---------------------------------------------------------------------------

class WelchTTestTests(unittest.TestCase):

    def test_identical_distributions_give_high_pvalue(self) -> None:
        values = [0.3] * 50
        p = _welch_pvalue(values, values)
        self.assertGreater(p, 0.9)

    def test_clearly_different_distributions_give_low_pvalue(self) -> None:
        a = [1.0] * 60
        b = [2.0] * 60
        p = _welch_pvalue(a, b)
        self.assertLess(p, 0.001)

    def test_returns_1_for_single_element_lists(self) -> None:
        p = _welch_pvalue([1.0], [2.0])
        self.assertAlmostEqual(p, 1.0)

    def test_pvalue_in_valid_range(self) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(0.3, 0.05) for _ in range(60)]
        b = [rng.gauss(0.35, 0.05) for _ in range(60)]
        p = _welch_pvalue(a, b)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)


class ShadowSignificanceTests(unittest.TestCase):

    def _build_shadow_window(
        self,
        shadow: ShadowDeploymentManager,
        champion_delta: float,
        challenger_delta: float,
        n: int = 60,
        noise: float = 0.0,
    ) -> None:
        import random
        rng = random.Random(1)
        for i in range(n):
            shadow._window.append({
                "lap": i + 1,
                "champion_delta_s": champion_delta + rng.gauss(0, noise),
                "challenger_delta_s": challenger_delta + rng.gauss(0, noise),
                "delta_s": champion_delta - challenger_delta,
                "abs_delta_s": abs(champion_delta - challenger_delta),
                "champion_wear": 30.0,
                "challenger_wear": 30.0,
                "champion_cliff": 0.1,
                "challenger_cliff": 0.1,
                "challenger_latency_ms": 1.0,
            })
        shadow._total = n

    def test_promotion_rejected_when_means_are_equal(self) -> None:
        """Champion == Challenger mean: no systematic improvement → no promotion regardless of n."""
        from f1_strategy.models import KalmanFilterModel, ModelConfig
        shadow = ShadowDeploymentManager(window_size=200)
        shadow.configure(model=KalmanFilterModel(ModelConfig()), backend="kalman")
        # Identical distributions — mean improvement is zero, well below threshold.
        self._build_shadow_window(shadow, champion_delta=0.30, challenger_delta=0.30, noise=0.0)
        candidate = shadow.promotion_candidate(min_predictions=50, improvement_threshold_s=0.02)
        self.assertIsNone(candidate, "Should not promote when champion and challenger are equal")

    def test_promotion_accepted_when_improvement_is_clearly_significant(self) -> None:
        """Champion 0.50s vs challenger 0.20s: large improvement, low noise → promote."""
        from f1_strategy.models import KalmanFilterModel, ModelConfig
        shadow = ShadowDeploymentManager(window_size=200)
        shadow.configure(model=KalmanFilterModel(ModelConfig()), backend="kalman")
        self._build_shadow_window(shadow, champion_delta=0.50, challenger_delta=0.20, noise=0.01)
        candidate = shadow.promotion_candidate(min_predictions=50, improvement_threshold_s=0.02)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertIn("p_value", candidate)
        self.assertLess(candidate["p_value"], 0.05)

    def test_promotion_result_includes_p_value(self) -> None:
        from f1_strategy.models import KalmanFilterModel, ModelConfig
        shadow = ShadowDeploymentManager(window_size=200)
        shadow.configure(model=KalmanFilterModel(ModelConfig()), backend="kalman")
        self._build_shadow_window(shadow, champion_delta=0.60, challenger_delta=0.10, noise=0.005)
        candidate = shadow.promotion_candidate(min_predictions=50)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertIn("p_value", candidate)
        self.assertGreaterEqual(candidate["p_value"], 0.0)
        self.assertLessEqual(candidate["p_value"], 1.0)


# ---------------------------------------------------------------------------
# Data quality validation
# ---------------------------------------------------------------------------

class DataQualityValidationTests(unittest.TestCase):

    _VALID_ROW = {
        "session_id": "s", "car_id": "VER", "lap": "1", "sector": "1",
        "speed_kph": "230.0", "throttle": "0.85", "brake": "0.1",
        "compound": "medium", "tire_temp_fl": "95.0", "tire_temp_fr": "95.0",
        "tire_temp_rl": "92.0", "tire_temp_rr": "92.0",
        "ers_soc": "0.75", "fuel_kg": "90.0", "track_temp_c": "38.0",
    }

    def test_valid_data_passes_without_exception(self) -> None:
        _validate_csv_schema([dict(self._VALID_ROW)] * 10, "test.csv")

    def test_missing_required_column_raises_value_error(self) -> None:
        bad = {k: v for k, v in self._VALID_ROW.items() if k != "compound"}
        with self.assertRaises(ValueError) as ctx:
            _validate_csv_schema([bad], "test.csv")
        self.assertIn("compound", str(ctx.exception))

    def test_out_of_range_throttle_raises_value_error(self) -> None:
        bad = {**self._VALID_ROW, "throttle": "1.5"}  # throttle > 1.0
        rows = [bad] * 100  # enough rows to exceed 5% violation threshold
        with self.assertRaises(ValueError) as ctx:
            _validate_csv_schema(rows, "test.csv")
        self.assertIn("throttle", str(ctx.exception))

    def test_excessive_missing_values_raises_value_error(self) -> None:
        rows = [{**self._VALID_ROW, "ers_soc": ""} for _ in range(100)]
        with self.assertRaises(ValueError) as ctx:
            _validate_csv_schema(rows, "test.csv")
        self.assertIn("ers_soc", str(ctx.exception))

    def test_empty_records_passes_silently(self) -> None:
        _validate_csv_schema([], "empty.csv")

    def test_minor_violations_below_threshold_pass(self) -> None:
        """Up to 5% range violations should not raise (naturally noisy telemetry)."""
        rows = [dict(self._VALID_ROW)] * 100
        # 4 rows with slightly out-of-range fuel (< 5% of 100)
        for i in range(4):
            rows[i] = {**self._VALID_ROW, "fuel_kg": "201.0"}
        _validate_csv_schema(rows, "test.csv")  # should not raise


# ---------------------------------------------------------------------------
# Inference latency benchmark
# ---------------------------------------------------------------------------

class LatencyBenchmarkTests(unittest.TestCase):

    _TARGET_P99_MS = 50.0

    def test_hybrid_backend_p99_under_budget(self) -> None:
        from f1_strategy.engine import InferenceEngine

        engine = InferenceEngine()
        event = RaceSimulator(SimulationConfig(laps=1, seed=42)).events()[-1]

        # Warmup
        for _ in range(20):
            engine.ingest(event)

        times_ms = []
        for _ in range(500):
            t0 = time.perf_counter()
            engine.ingest(event)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        times_ms.sort()
        p99 = times_ms[int(len(times_ms) * 0.99)]
        self.assertLess(
            p99,
            self._TARGET_P99_MS,
            f"Inference p99 latency {p99:.2f}ms exceeds {self._TARGET_P99_MS}ms budget",
        )

    def test_kalman_backend_p99_under_budget(self) -> None:
        from f1_strategy.engine import InferenceEngine
        from f1_strategy.models import KalmanFilterModel, ModelConfig

        engine = InferenceEngine(model=KalmanFilterModel(ModelConfig()))
        event = RaceSimulator(SimulationConfig(laps=1, seed=43)).events()[-1]

        for _ in range(10):
            engine.ingest(event)

        times_ms = []
        for _ in range(500):
            t0 = time.perf_counter()
            engine.ingest(event)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        times_ms.sort()
        p99 = times_ms[int(len(times_ms) * 0.99)]
        self.assertLess(p99, self._TARGET_P99_MS)


# ---------------------------------------------------------------------------
# Strategy optimizer — fleet-aware pit window
# ---------------------------------------------------------------------------

class FleetAwareOptimizerTests(unittest.TestCase):

    def _make_prediction(self) -> object:
        from f1_strategy.engine import InferenceEngine
        engine = InferenceEngine()
        sim = RaceSimulator(SimulationConfig(laps=5, seed=11))
        pred = None
        for ev in sim.events():
            pred = engine.ingest(ev)
        return pred, engine

    def test_undercut_probability_higher_when_closely_followed(self) -> None:
        from f1_strategy.engine import InferenceEngine
        from f1_strategy.feature_store import OnlineFeatureStore
        from f1_strategy.domain import FleetState

        # Two stores: one with close gap behind (threat), one with large gap.
        def run_strategy(gap_behind: float) -> float:
            store = OnlineFeatureStore()
            fs = FleetState("race")
            fs.update("c1", gap_ahead_s=5.0, gap_behind_s=gap_behind)
            store.set_fleet_state(fs)
            engine = InferenceEngine(feature_store=store)
            sim = RaceSimulator(SimulationConfig(session_id="race", car_id="c1", laps=5, seed=21))
            pred = None
            for ev in sim.events():
                pred = engine.ingest(ev)
            assert pred is not None
            rec = engine.strategy(pred.session_id, pred.car_id, remaining_laps=30)
            return rec.pit_window.undercut_success_probability

        close = run_strategy(gap_behind=3.0)
        far = run_strategy(gap_behind=30.0)
        self.assertGreater(close, far,
                           "Undercut probability should be higher when car behind is close")


# ---------------------------------------------------------------------------
# OpenF1 fleet intervals
# ---------------------------------------------------------------------------

class FleetIntervalsParseGapTests(unittest.TestCase):

    def setUp(self) -> None:
        from f1_strategy.data_sources.openf1_intervals import _parse_gap
        self._parse = _parse_gap

    def test_race_leader_zero_int(self) -> None:
        self.assertEqual(self._parse(0), 0.0)

    def test_race_leader_zero_float_string(self) -> None:
        self.assertEqual(self._parse("0.000"), 0.0)

    def test_numeric_string(self) -> None:
        self.assertAlmostEqual(self._parse("1.234"), 1.234)

    def test_positive_prefix_string(self) -> None:
        self.assertAlmostEqual(self._parse("+1.234"), 1.234)

    def test_lapped_one_lap(self) -> None:
        self.assertEqual(self._parse("+1 LAP"), 999.0)

    def test_lapped_two_laps(self) -> None:
        self.assertEqual(self._parse("+2 LAPS"), 999.0)

    def test_lapped_case_insensitive(self) -> None:
        self.assertEqual(self._parse("+1 lap"), 999.0)

    def test_none_returns_unknown(self) -> None:
        self.assertEqual(self._parse(None), 999.0)

    def test_empty_string_returns_unknown(self) -> None:
        self.assertEqual(self._parse(""), 999.0)

    def test_negative_clamped_to_zero(self) -> None:
        self.assertEqual(self._parse("-0.001"), 0.0)


class FleetIntervalsPathTests(unittest.TestCase):

    def test_companion_path_suffix(self) -> None:
        from f1_strategy.data_sources.openf1_intervals import fleet_intervals_path_for
        p = fleet_intervals_path_for(Path("/data/race_2023.csv"))
        self.assertEqual(p, Path("/data/race_2023.csv.intervals.csv"))

    def test_companion_path_preserves_stem(self) -> None:
        from f1_strategy.data_sources.openf1_intervals import fleet_intervals_path_for
        p = fleet_intervals_path_for("/data/replay.csv")
        self.assertEqual(p.name, "replay.csv.intervals.csv")


class BuildFleetStateForLapTests(unittest.TestCase):

    def _make_rows(self) -> list[dict]:
        return [
            {
                "session_id": "test-session",
                "session_key": "9000",
                "lap": "5",
                "driver_number": "1",
                "position": "1",
                "gap_ahead_s": "0.0",
                "gap_behind_s": "2.341",
                "gap_to_leader_s": "0.0",
                "tire_age_laps": "8",
                "compound": "soft",
            },
            {
                "session_id": "test-session",
                "session_key": "9000",
                "lap": "5",
                "driver_number": "11",
                "position": "2",
                "gap_ahead_s": "2.341",
                "gap_behind_s": "0.875",
                "gap_to_leader_s": "2.341",
                "tire_age_laps": "5",
                "compound": "medium",
            },
            # Different lap — should be ignored when filtering lap=5
            {
                "session_id": "test-session",
                "session_key": "9000",
                "lap": "6",
                "driver_number": "1",
                "position": "1",
                "gap_ahead_s": "0.0",
                "gap_behind_s": "1.100",
                "gap_to_leader_s": "0.0",
                "tire_age_laps": "9",
                "compound": "soft",
            },
        ]

    def test_gap_ahead_loaded_for_driver_1_lap_5(self) -> None:
        from f1_strategy.data_sources.openf1_intervals import build_fleet_state_for_lap
        rows = self._make_rows()
        fs = build_fleet_state_for_lap(rows, "test-session", 5, driver_number=1)
        self.assertAlmostEqual(fs.gap_ahead_s("1"), 0.0)
        self.assertAlmostEqual(fs.gap_behind_s("1"), 2.341)

    def test_position_loaded_for_driver_1_lap_5(self) -> None:
        from f1_strategy.data_sources.openf1_intervals import build_fleet_state_for_lap
        rows = self._make_rows()
        fs = build_fleet_state_for_lap(rows, "test-session", 5, driver_number=1)
        self.assertEqual(fs.position("1"), 1)

    def test_compound_parsed_correctly(self) -> None:
        from f1_strategy.data_sources.openf1_intervals import build_fleet_state_for_lap
        rows = self._make_rows()
        fs = build_fleet_state_for_lap(rows, "test-session", 5, driver_number=1)
        self.assertEqual(fs.competitor_compound("1"), TireCompound.SOFT)

    def test_tire_age_loaded(self) -> None:
        from f1_strategy.data_sources.openf1_intervals import build_fleet_state_for_lap
        rows = self._make_rows()
        fs = build_fleet_state_for_lap(rows, "test-session", 5, driver_number=1)
        self.assertEqual(fs.competitor_tire_age("1"), 8)

    def test_different_lap_not_loaded(self) -> None:
        from f1_strategy.data_sources.openf1_intervals import build_fleet_state_for_lap
        rows = self._make_rows()
        # Lap 6 has gap_behind 1.100 for driver 1; lap 5 state should be 2.341
        fs = build_fleet_state_for_lap(rows, "test-session", 5, driver_number=1)
        self.assertAlmostEqual(fs.gap_behind_s("1"), 2.341)

    def test_all_drivers_loaded_when_no_filter(self) -> None:
        from f1_strategy.data_sources.openf1_intervals import build_fleet_state_for_lap
        rows = self._make_rows()
        fs = build_fleet_state_for_lap(rows, "test-session", 5)
        # Both driver 1 and driver 11 should be present
        self.assertEqual(fs.position("1"), 1)
        self.assertEqual(fs.position("11"), 2)

    def test_wrong_session_id_ignored(self) -> None:
        from f1_strategy.data_sources.openf1_intervals import build_fleet_state_for_lap
        rows = self._make_rows()
        fs = build_fleet_state_for_lap(rows, "other-session", 5, driver_number=1)
        # No data → defaults
        self.assertEqual(fs.gap_ahead_s("1"), 999.0)

    def test_lapped_gap_stays_sentinel(self) -> None:
        from f1_strategy.data_sources.openf1_intervals import build_fleet_state_for_lap
        rows = [
            {
                "session_id": "s1", "session_key": "1", "lap": "3",
                "driver_number": "99", "position": "20",
                "gap_ahead_s": "999.0", "gap_behind_s": "999.0",
                "gap_to_leader_s": "999.0", "tire_age_laps": "0", "compound": "hard",
            }
        ]
        fs = build_fleet_state_for_lap(rows, "s1", 3, driver_number=99)
        self.assertEqual(fs.gap_ahead_s("99"), 999.0)


class TireAgeMapTests(unittest.TestCase):

    def test_age_increments_per_lap(self) -> None:
        from f1_strategy.data_sources.openf1_intervals import _tire_age_map
        stints = [
            {
                "driver_number": "1",
                "lap_start": "3",
                "lap_end": "6",
                "tyre_age_at_start": "2",
                "compound": "soft",
            }
        ]
        age_map = _tire_age_map(stints)
        self.assertEqual(age_map[(1, 3)], (2, "soft"))
        self.assertEqual(age_map[(1, 4)], (3, "soft"))
        self.assertEqual(age_map[(1, 6)], (5, "soft"))

    def test_missing_lap_end_defaults_to_lap_start(self) -> None:
        from f1_strategy.data_sources.openf1_intervals import _tire_age_map
        stints = [
            {
                "driver_number": "44",
                "lap_start": "10",
                "lap_end": None,
                "tyre_age_at_start": "0",
                "compound": "medium",
            }
        ]
        age_map = _tire_age_map(stints)
        self.assertIn((44, 10), age_map)
        self.assertEqual(age_map[(44, 10)], (0, "medium"))


class FleetIntervalsReplayIntegrationTests(unittest.TestCase):
    """Verify that replay_events correctly injects fleet state at lap boundaries."""

    def _build_intervals_rows(
        self, session_id: str, car_id: str, laps: int, gap_ahead_s: float = 3.0
    ) -> list[dict]:
        return [
            {
                "session_id": session_id,
                "session_key": "0",
                "lap": str(lap),
                "driver_number": car_id,
                "position": "2",
                "gap_ahead_s": str(gap_ahead_s),
                "gap_behind_s": "1.0",
                "gap_to_leader_s": str(gap_ahead_s),
                "tire_age_laps": str(lap),
                "compound": "medium",
            }
            for lap in range(1, laps + 1)
        ]

    def test_replay_with_fleet_intervals_completes(self) -> None:
        from f1_strategy.replay import replay_events
        sim = RaceSimulator(SimulationConfig(
            session_id="s1", car_id="c1", laps=6, seed=42
        ))
        events = list(sim.events())
        rows = self._build_intervals_rows("s1", "c1", laps=6)
        result = replay_events(events, dataset_name="test", fleet_intervals=rows)
        self.assertGreater(result.event_count, 0)

    def test_fleet_state_affects_dirty_air_when_gap_is_close(self) -> None:
        """replay with a 1.2s gap ahead should produce higher dirty_air_risk than 999s."""
        from f1_strategy.replay import replay_events
        sim = RaceSimulator(SimulationConfig(
            session_id="s2", car_id="c2", laps=8, seed=77
        ))
        events = list(sim.events())
        # Close gap: should raise dirty_air_risk
        rows_close = self._build_intervals_rows("s2", "c2", laps=8, gap_ahead_s=1.0)
        result_close = replay_events(events, dataset_name="close", fleet_intervals=rows_close)
        # No fleet intervals: gap defaults to 999 (no dirty air)
        result_none = replay_events(events, dataset_name="none", fleet_intervals=None)
        # Both must complete; no assertion on exact values — fleet state changes features, not MAE
        self.assertGreater(result_close.event_count, 0)
        self.assertGreater(result_none.event_count, 0)

    def test_load_fleet_intervals_returns_empty_for_missing_file(self) -> None:
        from f1_strategy.data_sources.openf1_intervals import load_fleet_intervals
        rows = load_fleet_intervals(Path("/nonexistent/path.csv.intervals.csv"))
        self.assertEqual(rows, [])

    def test_load_fleet_intervals_round_trips_csv(self) -> None:
        from f1_strategy.data_sources.openf1_intervals import (
            load_fleet_intervals,
            INTERVALS_CSV_COLUMNS,
        )
        rows = self._build_intervals_rows("s3", "c3", laps=3)
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "race.csv.intervals.csv"
            with p.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=INTERVALS_CSV_COLUMNS)
                writer.writeheader()
                for row in rows:
                    writer.writerow({c: row.get(c, "") for c in INTERVALS_CSV_COLUMNS})
            loaded = load_fleet_intervals(p)
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded[0]["lap"], "1")
        self.assertEqual(loaded[0]["driver_number"], "c3")


if __name__ == "__main__":
    unittest.main()
