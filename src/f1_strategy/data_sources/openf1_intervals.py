"""OpenF1 fleet intervals export — gap, position, and competitor tire state per lap.

Fetches /v1/intervals, /v1/position, /v1/stints, and /v1/laps for an entire
race session (all drivers), joins to per-lap fleet snapshots, and writes a
companion CSV next to the single-driver replay CSV.

Auto-detection: run_replay_evaluation() loads the companion when
{replay_csv}.intervals.csv exists alongside the replay dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from f1_strategy.data_sources.openf1_export import _find_session, _get, _parse_dt
from f1_strategy.domain import FleetState, TireCompound

FLEET_INTERVALS_SUFFIX = ".intervals.csv"
INTERVALS_CSV_COLUMNS = [
    "session_id", "session_key", "lap", "driver_number",
    "position", "gap_ahead_s", "gap_behind_s", "gap_to_leader_s",
    "tire_age_laps", "compound",
]

_LAPPED_S = 999.0    # sentinel value for "+1 LAP" etc.
_UNKNOWN_S = 999.0   # sentinel for missing gap data

_COMPOUND_MAP: dict[str, TireCompound] = {
    "soft": TireCompound.SOFT, "medium": TireCompound.MEDIUM,
    "hard": TireCompound.HARD, "intermediate": TireCompound.INTERMEDIATE,
    "wet": TireCompound.WET,
}


@dataclass(frozen=True)
class FleetIntervalsExportConfig:
    year: int
    event: str
    session: str
    output: Path


def fleet_intervals_path_for(replay_csv: Path | str) -> Path:
    """Return the companion fleet intervals CSV path for a given replay CSV."""
    p = Path(replay_csv)
    return p.with_suffix(p.suffix + FLEET_INTERVALS_SUFFIX)


# ── Gap parsing ───────────────────────────────────────────────────────────────

def _parse_gap(value: Any) -> float:
    """Parse an OpenF1 interval or gap_to_leader value to seconds.

    Handles: 0 (int/float leader), "0.000", "1.234", "+1 LAP", "+2 LAPS",
    None, "". Returns _LAPPED_S for lapped cars, _UNKNOWN_S for missing data.
    """
    if value is None or value == "":
        return _UNKNOWN_S
    s = str(value).strip()
    if "LAP" in s.upper():
        return _LAPPED_S
    try:
        return max(0.0, float(s))
    except (ValueError, TypeError):
        return _UNKNOWN_S


# ── API fetchers ──────────────────────────────────────────────────────────────

def _fetch_all_intervals(session_key: int) -> list[dict]:
    rows = _get("intervals", {"session_key": session_key})
    return sorted(rows, key=lambda r: r.get("date", ""))


def _fetch_all_positions(session_key: int) -> list[dict]:
    rows = _get("position", {"session_key": session_key})
    return sorted(rows, key=lambda r: r.get("date", ""))


def _fetch_all_stints(session_key: int) -> list[dict]:
    return _get("stints", {"session_key": session_key})


def _fetch_all_laps(session_key: int) -> list[dict]:
    rows = _get("laps", {"session_key": session_key})
    return sorted(rows, key=lambda r: (
        int(r.get("driver_number", 0)), int(r.get("lap_number", 0))
    ))


# ── Index builders ────────────────────────────────────────────────────────────

def _by_driver(rows: list[dict]) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = {}
    for row in rows:
        dn = int(row.get("driver_number", 0))
        result.setdefault(dn, []).append(row)
    return result


def _tire_age_map(stints: list[dict]) -> dict[tuple[int, int], tuple[int, str]]:
    """Return {(driver_number, lap_number): (tire_age_laps, compound_str)}."""
    out: dict[tuple[int, int], tuple[int, str]] = {}
    for s in stints:
        dn = int(s.get("driver_number", 0))
        lap_start = int(s.get("lap_start") or 1)
        lap_end = int(s.get("lap_end") or lap_start)
        age_at_start = int(s.get("tyre_age_at_start") or 0)
        compound = str(s.get("compound") or "MEDIUM").lower()
        for lap in range(lap_start, lap_end + 1):
            out[(dn, lap)] = (age_at_start + (lap - lap_start), compound)
    return out


# ── Per-lap lookups ───────────────────────────────────────────────────────────

def _latest_at(rows: list[dict], at: datetime) -> dict | None:
    """Last row whose 'date' <= at (rows must already be sorted ascending by date)."""
    best: dict | None = None
    for row in rows:
        row_dt = _parse_dt(row.get("date"))
        if row_dt is None:
            continue
        if row_dt <= at:
            best = row
        else:
            break
    return best


def _position_at(position_rows: list[dict], at: datetime) -> int:
    row = _latest_at(position_rows, at)
    return int(row["position"]) if row else 10


def _driver_at_position_at(
    positions_by_dn: dict[int, list[dict]],
    target_pos: int,
    at: datetime,
) -> int | None:
    """Return the driver_number holding target_pos at time at, or None."""
    best: tuple[float, int] | None = None
    for dn, rows in positions_by_dn.items():
        row = _latest_at(rows, at)
        if row is None:
            continue
        if int(row.get("position", -1)) == target_pos:
            row_dt = _parse_dt(row.get("date"))
            ts = row_dt.timestamp() if row_dt else 0.0
            if best is None or ts > best[0]:
                best = (ts, dn)
    return best[1] if best else None


# ── Main join ─────────────────────────────────────────────────────────────────

def _join_fleet_to_laps(
    intervals: list[dict],
    positions: list[dict],
    stints: list[dict],
    laps: list[dict],
    session_id: str,
    session_key: int,
) -> list[dict]:
    """Join all OpenF1 sources into one row per (driver, lap)."""
    intervals_by_dn = _by_driver(intervals)
    positions_by_dn = _by_driver(positions)
    laps_by_dn: dict[int, list[dict]] = {}
    for row in laps:
        dn = int(row.get("driver_number", 0))
        laps_by_dn.setdefault(dn, []).append(row)
    for dn in laps_by_dn:
        laps_by_dn[dn].sort(key=lambda r: int(r.get("lap_number", 0)))
    age_map = _tire_age_map(stints)

    rows: list[dict] = []
    for dn, driver_laps in laps_by_dn.items():
        dn_intervals = intervals_by_dn.get(dn, [])
        dn_positions = positions_by_dn.get(dn, [])

        for i, lap_row in enumerate(driver_laps):
            lap_number = int(lap_row.get("lap_number", 0))
            if lap_number <= 0:
                continue

            # Lap end time = start of next lap OR start + duration
            lap_end_dt: datetime | None = None
            if i + 1 < len(driver_laps):
                lap_end_dt = _parse_dt(driver_laps[i + 1].get("date_start"))
            if lap_end_dt is None:
                start_dt = _parse_dt(lap_row.get("date_start"))
                dur = lap_row.get("lap_duration")
                if start_dt and dur:
                    lap_end_dt = start_dt + timedelta(seconds=float(dur))
            if lap_end_dt is None:
                continue

            # Gap ahead from this driver's own interval reading
            gap_row = _latest_at(dn_intervals, lap_end_dt)
            gap_ahead_s = _parse_gap(gap_row.get("interval")) if gap_row else _UNKNOWN_S
            gap_to_leader_s = _parse_gap(gap_row.get("gap_to_leader")) if gap_row else _UNKNOWN_S

            # Position at lap end
            pos = _position_at(dn_positions, lap_end_dt)

            # Gap behind = interval of the car at position+1 at the same moment
            car_behind = _driver_at_position_at(positions_by_dn, pos + 1, lap_end_dt)
            if car_behind is not None:
                behind_row = _latest_at(intervals_by_dn.get(car_behind, []), lap_end_dt)
                gap_behind_s = _parse_gap(behind_row.get("interval")) if behind_row else _UNKNOWN_S
            else:
                gap_behind_s = _UNKNOWN_S

            tire_age, compound = age_map.get((dn, lap_number), (0, "medium"))

            rows.append({
                "session_id": session_id,
                "session_key": session_key,
                "lap": lap_number,
                "driver_number": dn,
                "position": pos,
                "gap_ahead_s": round(gap_ahead_s, 3) if gap_ahead_s < _LAPPED_S else _LAPPED_S,
                "gap_behind_s": round(gap_behind_s, 3) if gap_behind_s < _LAPPED_S else _LAPPED_S,
                "gap_to_leader_s": round(gap_to_leader_s, 3) if gap_to_leader_s < _LAPPED_S else _LAPPED_S,
                "tire_age_laps": tire_age,
                "compound": compound,
            })

    return sorted(rows, key=lambda r: (r["lap"], r["position"]))


# ── Export ────────────────────────────────────────────────────────────────────

def export_fleet_intervals(config: FleetIntervalsExportConfig) -> dict[str, Any]:
    """Export per-lap fleet gap and position data for an entire session to CSV."""
    print(f"[openf1-fleet] Fetching {config.year} {config.event} {config.session} …")
    session_row = _find_session(config.year, config.event, config.session)
    session_key = int(session_row["session_key"])
    circuit = str(
        session_row.get("circuit_short_name")
        or session_row.get("location")
        or config.event
    )
    circuit_slug = circuit.strip().lower().replace(" ", "-")
    session_id = f"openf1-{config.year}-{circuit_slug}-{config.session.lower()}"
    print(f"  session_key={session_key}  id={session_id}")

    print("  → intervals …")
    intervals = _fetch_all_intervals(session_key)
    print(f"  → positions … ({len(intervals)} interval readings)")
    positions = _fetch_all_positions(session_key)
    print(f"  → stints … ({len(positions)} position readings)")
    stints = _fetch_all_stints(session_key)
    print(f"  → laps … ({len(stints)} stint records)")
    laps = _fetch_all_laps(session_key)
    print(f"  Joining … ({len(laps)} lap records)")

    rows = _join_fleet_to_laps(intervals, positions, stints, laps, session_id, session_key)
    lap_count = len({r["lap"] for r in rows})
    driver_count = len({r["driver_number"] for r in rows})
    print(f"  Joined: {len(rows)} rows  {driver_count} drivers  {lap_count} laps")

    config.output.parent.mkdir(parents=True, exist_ok=True)
    with config.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INTERVALS_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in INTERVALS_CSV_COLUMNS})

    manifest: dict[str, Any] = {
        "source": "openf1-fleet-intervals",
        "generated_at": datetime.now(UTC).isoformat(),
        "year": config.year,
        "event": config.event,
        "session": config.session,
        "session_key": session_key,
        "session_id": session_id,
        "circuit_slug": circuit_slug,
        "output": str(config.output),
        "row_count": len(rows),
        "lap_count": lap_count,
        "driver_count": driver_count,
        "columns": INTERVALS_CSV_COLUMNS,
    }
    manifest_path = config.output.with_suffix(config.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Written: {config.output}")
    return manifest


# ── Loading & FleetState construction ─────────────────────────────────────────

def load_fleet_intervals(path: Path | str) -> list[dict]:
    """Load a fleet intervals CSV.  Returns [] if file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def build_fleet_state_for_lap(
    rows: list[dict],
    session_id: str,
    lap: int,
    *,
    driver_number: int | str | None = None,
) -> FleetState:
    """Build a FleetState snapshot from fleet interval rows for one lap.

    Populates all drivers found for (session_id, lap).  When driver_number is
    given only that driver is loaded — use this during single-driver replay to
    avoid building state for the full 20-car grid on every lap.
    """
    fs = FleetState(session_id)
    target_dn = str(driver_number) if driver_number is not None else None

    for row in rows:
        if row.get("session_id") != session_id:
            continue
        try:
            row_lap = int(row.get("lap", -1))
        except (ValueError, TypeError):
            continue
        if row_lap != lap:
            continue

        dn_str = str(row.get("driver_number", ""))
        if target_dn is not None and dn_str != target_dn:
            continue

        try:
            gap_ahead = float(row.get("gap_ahead_s", _UNKNOWN_S))
            gap_behind = float(row.get("gap_behind_s", _UNKNOWN_S))
            pos = int(float(row.get("position", 10)))
            tire_age = int(float(row.get("tire_age_laps", 0)))
            compound = _COMPOUND_MAP.get(
                str(row.get("compound", "medium")).lower(), TireCompound.MEDIUM
            )
        except (ValueError, TypeError):
            continue

        fs.update(
            dn_str,
            position=pos,
            gap_ahead_s=gap_ahead,
            gap_behind_s=gap_behind,
            competitor_tire_age=tire_age,
            competitor_compound=compound,
        )

    return fs


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export OpenF1 fleet gap and position data for an entire race session. "
            "Output is a companion CSV that replay evaluation picks up automatically."
        )
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--event", required=True, help="Country or circuit name (e.g. Austin)")
    parser.add_argument("--session", required=True, help="Race | Qualifying | FP1 …")
    parser.add_argument("--output", required=True, help="Path for the output CSV")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = export_fleet_intervals(
        FleetIntervalsExportConfig(
            year=args.year,
            event=args.event,
            session=args.session,
            output=Path(args.output),
        )
    )
    print(
        f"[openf1-fleet] {manifest['row_count']} rows  "
        f"{manifest['driver_count']} drivers  "
        f"{manifest['lap_count']} laps → {manifest['output']}"
    )


if __name__ == "__main__":
    main()
