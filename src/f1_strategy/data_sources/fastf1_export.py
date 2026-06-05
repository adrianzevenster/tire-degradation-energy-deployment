from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from f1_strategy.domain import TireCompound
from f1_strategy.replay import (
    REPLAY_OPTIONAL_COLUMNS,
    REPLAY_REQUIRED_COLUMNS,
    dataset_fingerprint,
    validate_replay_payload,
)


REPLAY_COLUMNS = REPLAY_REQUIRED_COLUMNS + REPLAY_OPTIONAL_COLUMNS

FIELD_PROVENANCE = {
    "session_id": "derived: export session identifier",
    "car_id": "observed: FastF1 driver number or driver code",
    "lap": "observed: FastF1 lap number",
    "sector": "derived: sector bucket from sector timing or telemetry distance thirds",
    "speed_kph": "observed: FastF1 car telemetry Speed",
    "throttle": "observed: FastF1 car telemetry Throttle, normalized to 0-1",
    "brake": "observed: FastF1 car telemetry Brake, normalized to 0-1",
    "steering_angle": "derived: heading curvature proxy when position is available, otherwise 0",
    "tire_temp_fl": "derived: public F1 data does not expose tire temperatures",
    "tire_temp_fr": "derived: public F1 data does not expose tire temperatures",
    "tire_temp_rl": "derived: public F1 data does not expose tire temperatures",
    "tire_temp_rr": "derived: public F1 data does not expose tire temperatures",
    "brake_temp": "derived: public F1 data does not expose brake temperatures",
    "slip_angle": "derived: public F1 data does not expose slip angle",
    "lateral_g": "derived: curvature/speed proxy when position is available",
    "ers_soc": "derived: public F1 data does not expose ERS state of charge",
    "ers_deployment_kw": "derived: public F1 data does not expose ERS deployment",
    "fuel_kg": "derived: fuel burn estimate from race lap index",
    "track_temp_c": "observed: FastF1 weather TrackTemp when available",
    "air_temp_c": "observed: FastF1 weather AirTemp when available",
    "humidity": "observed: FastF1 weather Humidity when available",
    "compound": "observed: FastF1 lap Compound when available",
    "lap_time_s": "observed: FastF1 lap time label",
    "timestamp_ms": "derived: FastF1 telemetry time when available, otherwise lap/sector index",
}


@dataclass(frozen=True)
class FastF1ExportConfig:
    year: int
    event: str
    session: str
    driver: str
    output: Path
    cache_dir: Path | None = None
    max_laps: int | None = None
    include_manifest: bool = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a FastF1 session into this project's replay CSV schema."
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--event",
        required=True,
        help="Grand Prix name or round number accepted by FastF1.",
    )
    parser.add_argument("--session", required=True, help="FastF1 session name, e.g. R, Q, FP2.")
    parser.add_argument("--driver", required=True, help="Driver code or number, e.g. VER or 1.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-laps", type=int, default=None)
    parser.add_argument("--no-manifest", action="store_true")
    return parser


def export_fastf1_replay(config: FastF1ExportConfig) -> dict[str, Any]:
    try:
        import fastf1
    except ImportError as exc:
        raise RuntimeError('Install FastF1 first: pip install -e ".[data]"') from exc

    if config.cache_dir is not None:
        config.cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(config.cache_dir))

    session = fastf1.get_session(config.year, _event_arg(config.event), config.session)
    session.load(laps=True, telemetry=True, weather=True, messages=False)
    rows = rows_from_fastf1_session(
        session,
        year=config.year,
        event=config.event,
        session_name=config.session,
        driver=config.driver,
        max_laps=config.max_laps,
    )
    if not rows:
        raise RuntimeError(
            f"No FastF1 replay rows exported for {config.year} {config.event} "
            f"{config.session} driver={config.driver}"
        )

    config.output.parent.mkdir(parents=True, exist_ok=True)
    write_replay_csv(config.output, rows)
    manifest = export_manifest(
        config=config,
        rows=rows,
        fastf1_version=getattr(fastf1, "__version__", "unknown"),
    )
    if config.include_manifest:
        _write_json(manifest_path_for(config.output), manifest)
    return manifest


def rows_from_fastf1_session(
    session: Any,
    *,
    year: int,
    event: str,
    session_name: str,
    driver: str,
    max_laps: int | None = None,
) -> list[dict[str, Any]]:
    laps = _selected_laps(session, driver)
    weather = _records(getattr(session, "weather_data", []))
    session_id = _session_id(year, event, session_name)
    circuit = str(event).strip().lower().replace(" ", "-")
    rows: list[dict[str, Any]] = []
    for lap in laps[:max_laps]:
        lap_number = int(_value(lap, "LapNumber", default=len(rows) // 3 + 1) or 1)
        lap_records = _lap_telemetry_records(lap)
        weather_row = _weather_for_lap(weather, lap)
        compound = _compound(_value(lap, "Compound", default="medium"))
        car_id = str(
            _value(lap, "DriverNumber", default=None)
            or _value(lap, "Driver", default=None)
            or driver
        )
        lap_time_s = _seconds(_value(lap, "LapTime", default=None))
        lap_duration_s = lap_time_s or sum(
            value
            for value in (
                _seconds(_value(lap, "Sector1Time", default=None)),
                _seconds(_value(lap, "Sector2Time", default=None)),
                _seconds(_value(lap, "Sector3Time", default=None)),
            )
            if value is not None
        )
        if not lap_duration_s:
            lap_duration_s = 90.0

        for sector in (1, 2, 3):
            sector_records = _sector_records(lap_records, sector)
            summary = _telemetry_summary(sector_records)
            payload = _replay_row(
                session_id=session_id,
                car_id=car_id,
                lap_number=lap_number,
                sector=sector,
                compound=compound,
                lap_time_s=lap_time_s,
                lap_duration_s=lap_duration_s,
                telemetry=summary,
                weather=weather_row,
                circuit=circuit,
            )
            validate_replay_payload(payload)
            rows.append(payload)
    return rows


def write_replay_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPLAY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in REPLAY_COLUMNS})


def export_manifest(
    *,
    config: FastF1ExportConfig,
    rows: list[dict[str, Any]],
    fastf1_version: str,
) -> dict[str, Any]:
    dataset_path = str(config.output)
    labeled_lap_times = [
        float(row["lap_time_s"])
        for row in rows
        if row.get("lap_time_s") is not None
    ]
    return {
        "source": "fastf1",
        "fastf1_version": fastf1_version,
        "generated_at": datetime.now(UTC).isoformat(),
        "year": config.year,
        "event": config.event,
        "session": config.session,
        "driver": config.driver,
        "output": dataset_path,
        "dataset_fingerprint": dataset_fingerprint(config.output) if config.output.exists() else "",
        "row_count": len(rows),
        "lap_count": len({row["lap"] for row in rows}),
        "session_count": len({row["session_id"] for row in rows}),
        "reference_lap_time_s": mean(labeled_lap_times) if labeled_lap_times else 90.0,
        "field_provenance": FIELD_PROVENANCE,
        "limitations": [
            "Public FastF1 data does not expose tire temperatures, brake temperatures, "
            "ERS state, fuel load, true tire wear, or slip angle.",
            "Unavailable private channels are deterministic proxies and should not be "
            "treated as observed labels.",
            "Use the manifest field_provenance map when deciding which replay fields "
            "are suitable for model evaluation gates.",
        ],
    }


def manifest_path_for(output: str | Path) -> Path:
    path = Path(output)
    return path.with_suffix(path.suffix + ".manifest.json")


def _replay_row(
    *,
    session_id: str,
    car_id: str,
    lap_number: int,
    sector: int,
    compound: TireCompound,
    lap_time_s: float | None,
    lap_duration_s: float,
    telemetry: dict[str, float],
    weather: dict[str, Any],
    circuit: str = "",
) -> dict[str, Any]:
    track_temp = _number(weather.get("TrackTemp"), 37.0)
    air_temp = _number(weather.get("AirTemp"), 27.0)
    humidity = _number(weather.get("Humidity"), 45.0) / 100.0
    speed = telemetry["speed_kph"]
    throttle = telemetry["throttle"]
    brake = telemetry["brake"]
    lateral_g = telemetry["lateral_g"]
    slip_angle = max(0.4, abs(lateral_g) * 0.55 + brake * 2.8 + (1.0 - throttle) * 0.35)
    tire_age = max(0, lap_number - 1)
    compound_heat = {
        TireCompound.SOFT: 3.5,
        TireCompound.MEDIUM: 1.8,
        TireCompound.HARD: 0.5,
        TireCompound.INTERMEDIATE: -2.0,
        TireCompound.WET: -4.0,
    }[compound]
    thermal_load = (
        track_temp * 0.34
        + brake * 8.0
        + throttle * 3.5
        + slip_angle * 1.7
        + tire_age * 0.18
    )
    tire_base = 66.0 + compound_heat + thermal_load
    brake_temp = 360.0 + brake * 470.0 + max(0.0, speed - 180.0) * 0.8
    ers_deployment = max(0.0, min(120.0, throttle * 92.0 - brake * 35.0 + speed * 0.06))
    ers_soc = max(0.05, min(1.0, 0.86 - lap_number * 0.012 + (3 - sector) * 0.018))
    fuel = max(0.0, 105.0 - (lap_number - 1) * 1.72)
    timestamp = int(telemetry.get("timestamp_ms") or (lap_number * 10_000 + sector * 3_000))
    return {
        "session_id": session_id,
        "car_id": car_id,
        "lap": lap_number,
        "sector": sector,
        "speed_kph": speed,
        "throttle": throttle,
        "brake": brake,
        "steering_angle": telemetry["steering_angle"],
        "tire_temp_fl": tire_base + lateral_g * 1.8 + sector * 0.4,
        "tire_temp_fr": tire_base - lateral_g * 1.2 + sector * 0.4,
        "tire_temp_rl": tire_base - 2.4 + throttle * 1.8,
        "tire_temp_rr": tire_base - 2.0 + throttle * 1.6,
        "brake_temp": brake_temp,
        "slip_angle": slip_angle,
        "lateral_g": abs(lateral_g),
        "ers_soc": ers_soc,
        "ers_deployment_kw": ers_deployment,
        "fuel_kg": fuel,
        "track_temp_c": track_temp,
        "air_temp_c": air_temp,
        "humidity": max(0.0, min(1.0, humidity)),
        "compound": compound.value,
        "lap_time_s": lap_time_s or lap_duration_s,
        "timestamp_ms": timestamp,
        "circuit": circuit,
    }


def _selected_laps(session: Any, driver: str) -> list[Any]:
    laps = getattr(session, "laps", [])
    if hasattr(laps, "pick_driver"):
        try:
            laps = laps.pick_driver(driver)
        except Exception:
            pass
    records = []
    if hasattr(laps, "iterlaps"):
        records = [lap for _, lap in laps.iterlaps()]
    elif hasattr(laps, "to_dict"):
        records = _records(laps)
    else:
        records = list(laps)
    filtered = [
        lap
        for lap in records
        if str(_value(lap, "Driver", default=driver)).lower() == driver.lower()
        or str(_value(lap, "DriverNumber", default=driver)) == str(driver)
    ]
    return filtered or records


def _lap_telemetry_records(lap: Any) -> list[dict[str, Any]]:
    for method in ("get_car_data", "get_telemetry"):
        getter = getattr(lap, method, None)
        if getter is None:
            continue
        try:
            telemetry = getter()
            if hasattr(telemetry, "add_distance"):
                telemetry = telemetry.add_distance()
            records = _records(telemetry)
            if records:
                return records
        except Exception:
            continue
    return []


def _records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        try:
            records = value.to_dict("records")
            return [dict(record) for record in records]
        except TypeError:
            pass
    if isinstance(value, dict):
        return [dict(value)]
    return [dict(item) for item in value]


def _sector_records(records: list[dict[str, Any]], sector: int) -> list[dict[str, Any]]:
    if not records:
        return []
    distances = [_number(record.get("Distance"), float("nan")) for record in records]
    finite_distances = [value for value in distances if value == value]
    if finite_distances and max(finite_distances) > min(finite_distances):
        lower = (
            min(finite_distances)
            + (sector - 1) * (max(finite_distances) - min(finite_distances)) / 3
        )
        upper = min(finite_distances) + sector * (max(finite_distances) - min(finite_distances)) / 3
        selected = [
            record
            for record in records
            if lower <= _number(record.get("Distance"), -1.0) <= upper
        ]
        return selected or records
    chunk = max(1, len(records) // 3)
    start = (sector - 1) * chunk
    end = len(records) if sector == 3 else sector * chunk
    return records[start:end] or records


def _telemetry_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    speed = _mean_field(records, "Speed", 235.0)
    throttle = _normalize_unit(_mean_field(records, "Throttle", 72.0))
    brake = _normalize_unit(_mean_field(records, "Brake", 0.0))
    steering = _steering_proxy(records)
    lateral_g = _lateral_g_proxy(records, speed)
    timestamp = _timestamp_ms(records[0]) if records else 0
    return {
        "speed_kph": speed,
        "throttle": throttle,
        "brake": brake,
        "steering_angle": steering,
        "lateral_g": lateral_g,
        "timestamp_ms": timestamp,
    }


def _mean_field(records: list[dict[str, Any]], field: str, default: float) -> float:
    values = [_number(record.get(field), float("nan")) for record in records]
    finite = [value for value in values if value == value]
    return mean(finite) if finite else default


def _steering_proxy(records: list[dict[str, Any]]) -> float:
    if len(records) < 3:
        return 0.0
    first = records[0]
    last = records[-1]
    x0 = _number(first.get("X"), float("nan"))
    y0 = _number(first.get("Y"), float("nan"))
    x1 = _number(last.get("X"), float("nan"))
    y1 = _number(last.get("Y"), float("nan"))
    if not all(value == value for value in (x0, y0, x1, y1)):
        return 0.0
    from math import atan2, degrees

    return max(-30.0, min(30.0, degrees(atan2(y1 - y0, x1 - x0)) / 3.0))


def _lateral_g_proxy(records: list[dict[str, Any]], speed_kph: float) -> float:
    steering = abs(_steering_proxy(records))
    return max(0.0, min(5.5, 0.7 + steering / 12.0 + max(0.0, speed_kph - 220.0) / 95.0))


def _weather_for_lap(weather_rows: list[dict[str, Any]], lap: Any) -> dict[str, Any]:
    if not weather_rows:
        return {"TrackTemp": 37.0, "AirTemp": 27.0, "Humidity": 45.0}
    lap_time = _milliseconds(_value(lap, "Time", default=None))
    if lap_time is None:
        return weather_rows[-1]
    candidates = [
        row for row in weather_rows if (_milliseconds(row.get("Time")) or 0) <= lap_time
    ]
    return candidates[-1] if candidates else weather_rows[0]


def _compound(value: Any) -> TireCompound:
    normalized = str(value or "medium").strip().lower()
    mapping = {
        "soft": TireCompound.SOFT,
        "medium": TireCompound.MEDIUM,
        "hard": TireCompound.HARD,
        "intermediate": TireCompound.INTERMEDIATE,
        "wet": TireCompound.WET,
    }
    return mapping.get(normalized, TireCompound.MEDIUM)


def _value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    try:
        return row[name]
    except Exception:
        return getattr(row, name, default)


def _number(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _normalize_unit(value: float) -> float:
    if value > 1.0:
        value = value / 100.0
    return max(0.0, min(1.0, value))


def _seconds(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _timestamp_ms(record: dict[str, Any]) -> int:
    for key in ("Time", "Date"):
        value = record.get(key)
        millis = _milliseconds(value)
        if millis is not None:
            return millis
    return 0


def _milliseconds(value: Any) -> int | None:
    if value is None:
        return None
    if hasattr(value, "total_seconds"):
        return int(value.total_seconds() * 1000)
    if hasattr(value, "timestamp"):
        return int(value.timestamp() * 1000)
    try:
        return int(float(value) * 1000)
    except (TypeError, ValueError):
        return None


def _session_id(year: int, event: str, session_name: str) -> str:
    normalized_event = str(event).strip().lower().replace(" ", "-")
    normalized_session = str(session_name).strip().lower().replace(" ", "-")
    return f"fastf1-{year}-{normalized_event}-{normalized_session}"


def _event_arg(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    manifest = export_fastf1_replay(
        FastF1ExportConfig(
            year=args.year,
            event=args.event,
            session=args.session,
            driver=args.driver,
            output=Path(args.output),
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            max_laps=args.max_laps,
            include_manifest=not args.no_manifest,
        )
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
