"""Integration tests against the live OpenF1 public API.

Run with:  pytest tests/test_integration_openf1.py -m integration -v

These tests hit the real OpenF1 endpoint and are excluded from the standard
CI suite (which only runs unit + system tests). They verify that:
  - The API contract has not changed (expected fields present)
  - The interval parsing pipeline handles live data correctly
  - Fleet-state construction works end-to-end with real session data
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
import pytest


OPENF1_BASE = "https://api.openf1.org/v1"


def _get(path: str, params: dict | None = None) -> list[dict]:
    """Minimal HTTP GET returning parsed JSON list."""
    url = f"{OPENF1_BASE}{path}"
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{query}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        pytest.skip(f"OpenF1 returned {exc.code} for {url} — likely rate-limited or data unavailable")
    except OSError as exc:
        pytest.skip(f"OpenF1 unreachable: {exc}")


@pytest.mark.integration
def test_openf1_sessions_endpoint_returns_data():
    """The /sessions endpoint returns a non-empty list with expected fields."""
    data = _get("/sessions", {"year": 2024, "session_name": "Race", "circuit_short_name": "Bahrain"})
    assert data, "Expected at least one session for 2024 Bahrain Race"
    session = data[0]
    assert "session_key" in session, f"Missing session_key in: {session}"
    assert "year" in session
    assert session["year"] == 2024


@pytest.mark.integration
def test_openf1_intervals_returns_expected_fields():
    """The /intervals endpoint returns records with gap and position fields."""
    sessions = _get("/sessions", {"year": 2024, "session_name": "Race", "circuit_short_name": "Bahrain"})
    if not sessions:
        pytest.skip("No 2024 Bahrain Race session found")
    session_key = sessions[0]["session_key"]
    intervals = _get("/intervals", {"session_key": session_key, "driver_number": 1})
    if not intervals:
        pytest.skip("No interval data for driver 1 in 2024 Bahrain")
    for record in intervals[:5]:
        assert "gap_to_leader" in record or "gap_to_leader_s" in record or "interval" in record, (
            f"Unexpected interval record shape: {list(record.keys())}"
        )


@pytest.mark.integration
def test_openf1_positions_returns_expected_fields():
    """The /position endpoint returns records with driver_number and position."""
    sessions = _get("/sessions", {"year": 2024, "session_name": "Race", "circuit_short_name": "Bahrain"})
    if not sessions:
        pytest.skip("No 2024 Bahrain Race session found")
    session_key = sessions[0]["session_key"]
    positions = _get("/position", {"session_key": session_key, "driver_number": 1})
    if not positions:
        pytest.skip("No position data")
    record = positions[0]
    assert "position" in record, f"Missing 'position' in: {list(record.keys())}"
    assert "driver_number" in record


@pytest.mark.integration
def test_openf1_stints_returns_compound_and_age():
    """The /stints endpoint returns compound and tire age for a known session."""
    sessions = _get("/sessions", {"year": 2024, "session_name": "Race", "circuit_short_name": "Bahrain"})
    if not sessions:
        pytest.skip("No session found")
    session_key = sessions[0]["session_key"]
    stints = _get("/stints", {"session_key": session_key, "driver_number": 1})
    if not stints:
        pytest.skip("No stint data")
    for stint in stints[:3]:
        assert "compound" in stint, f"Missing 'compound': {list(stint.keys())}"
        assert "lap_start" in stint or "lap_number" in stint, f"Missing lap field: {list(stint.keys())}"


@pytest.mark.integration
def test_fleet_intervals_export_pipeline():
    """Full export pipeline: fetch live intervals, build FleetState, check fields."""
    from f1_strategy.data_sources.openf1_intervals import (
        FleetIntervalsExportConfig,
        export_fleet_intervals,
        build_fleet_state_for_lap,
    )
    import tempfile, os

    sessions = _get("/sessions", {"year": 2024, "session_name": "Race", "circuit_short_name": "Bahrain"})
    if not sessions:
        pytest.skip("No session found")
    session_key = sessions[0]["session_key"]

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        config = FleetIntervalsExportConfig(
            session_key=session_key,
            year=2024,
            event="bahrain",
            session="R",
            output=tmp_path,
        )
        result = export_fleet_intervals(config)
        assert result.get("rows", 0) > 0, f"Expected rows, got: {result}"

        from f1_strategy.data_sources.openf1_intervals import load_fleet_intervals
        rows = load_fleet_intervals(tmp_path)
        assert rows, "No rows loaded from export"

        lap = rows[0]["lap"]
        session_id = rows[0]["session_id"]
        fleet = build_fleet_state_for_lap(rows, session_id, lap)
        assert fleet is not None
    finally:
        os.unlink(tmp_path)


@pytest.mark.integration
def test_openf1_laps_endpoint():
    """The /laps endpoint returns lap_duration and driver_number."""
    sessions = _get("/sessions", {"year": 2024, "session_name": "Race", "circuit_short_name": "Bahrain"})
    if not sessions:
        pytest.skip("No session found")
    session_key = sessions[0]["session_key"]
    laps = _get("/laps", {"session_key": session_key, "driver_number": 1})
    if not laps:
        pytest.skip("No lap data")
    record = laps[0]
    assert "lap_duration" in record or "lap_time" in record, f"No lap time field in: {list(record.keys())}"
    assert "lap_number" in record or "lap" in record, f"No lap number field in: {list(record.keys())}"
