"""
Live and replay telemetry streaming.

Two sources share a common LiveSource interface:

  ReplayStreamSource   Streams a historical FastF1 CSV at configurable speed.
                       Works immediately — no race weekend required.

  FastF1LiveSource     Connects to the F1 live timing SignalR feed during a
                       session and derives TelemetryEvents in near-real-time.
                       Requires an active F1 session and optionally an F1
                       account for authenticated access to car-data channels.

Both call an on_event callback with TelemetryEvent objects so the caller
(LiveStreamManager) is source-agnostic.
"""

from __future__ import annotations

import base64
import csv
import json
import queue
import threading
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from f1_strategy.domain import TelemetryEvent, TireCompound

# ── CarData.z SignalR channel indices ────────────────────────────────────────
_CH_SPEED = "2"
_CH_THROTTLE = "4"
_CH_BRAKE = "5"

_COMPOUND_MAP = {
    "soft": TireCompound.SOFT,
    "medium": TireCompound.MEDIUM,
    "hard": TireCompound.HARD,
    "intermediate": TireCompound.INTERMEDIATE,
    "wet": TireCompound.WET,
}


@dataclass
class StreamStatus:
    mode: str = "idle"          # "replay" | "live" | "idle"
    connected: bool = False
    session_id: str = ""
    driver: str = ""
    events_ingested: int = 0
    events_per_second: float = 0.0
    latest_lap: int = 0
    latest_lap_time_s: float | None = None
    current_compound: str = "medium"
    dataset_path: str = ""
    message_count: int = 0
    speed_multiplier: float = 1.0
    error: str = ""
    progress_pct: float = 0.0


# ── Replay source ─────────────────────────────────────────────────────────────

class ReplayStreamSource:
    """Streams a FastF1 replay CSV at real-time pace (or faster)."""

    def __init__(
        self,
        dataset_path: str,
        speed_multiplier: float = 1.0,
        on_event: Callable[[TelemetryEvent], None] | None = None,
    ) -> None:
        self._path = Path(dataset_path)
        self._speed = max(0.1, speed_multiplier)
        self._on_event = on_event
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.status = StreamStatus(
            mode="replay",
            dataset_path=str(dataset_path),
            speed_multiplier=speed_multiplier,
        )
        self._lock = threading.Lock()
        self._t_start: float = 0.0
        self._event_times: list[float] = []

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        self._t_start = time.monotonic()
        try:
            rows = self._load_rows()
            total = max(len(rows), 1)
            for index, row in enumerate(rows):
                if self._stop.is_set():
                    break
                event = _row_to_event(row)
                if event is None:
                    continue
                self._sleep_to_timestamp(row, index, rows)
                if self._on_event:
                    self._on_event(event)
                with self._lock:
                    self.status.events_ingested += 1
                    self.status.latest_lap = event.lap
                    self.status.latest_lap_time_s = event.lap_time_s
                    self.status.current_compound = event.compound.value
                    self.status.session_id = event.session_id
                    self.status.driver = event.car_id
                    self.status.progress_pct = round((index + 1) / total * 100, 1)
                    self._event_times.append(time.monotonic())
                    self._event_times = [t for t in self._event_times if t > time.monotonic() - 5]
                    self.status.events_per_second = len(self._event_times) / 5.0
            with self._lock:
                self.status.connected = False
                self.status.progress_pct = 100.0 if not self._stop.is_set() else self.status.progress_pct
        except Exception as exc:
            with self._lock:
                self.status.error = str(exc)
                self.status.connected = False

    def _load_rows(self) -> list[dict]:
        with self._path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        rows.sort(key=lambda r: int(r.get("timestamp_ms") or 0))
        with self._lock:
            self.status.connected = True
        return rows

    def _sleep_to_timestamp(self, row: dict, index: int, rows: list[dict]) -> None:
        if index == 0:
            return
        prev_ts = int(rows[index - 1].get("timestamp_ms") or 0)
        curr_ts = int(row.get("timestamp_ms") or 0)
        gap_ms = curr_ts - prev_ts
        if 0 < gap_ms < 10_000:
            time.sleep(gap_ms / 1000.0 / self._speed)


# ── FastF1 live source ────────────────────────────────────────────────────────

@dataclass
class _DriverState:
    session_id: str
    driver: str
    lap: int = 1
    sector: int = 1
    speed_kph: float = 200.0
    throttle: float = 0.70
    brake: float = 0.05
    steering_angle: float = 0.0
    lateral_g: float = 1.8
    slip_angle: float = 2.0
    track_temp_c: float = 37.0
    air_temp_c: float = 27.0
    humidity: float = 0.45
    compound: TireCompound = TireCompound.MEDIUM
    ers_soc: float = 0.80
    ers_deployment_kw: float = 60.0
    fuel_kg: float = 100.0
    tire_temp_base: float = 88.0
    brake_temp: float = 420.0
    latest_lap_time_s: float | None = None
    car_data_samples: int = 0
    _emit_interval: int = field(default=25, repr=False)

    def update_car_data(self, channels: dict) -> None:
        if _CH_SPEED in channels:
            self.speed_kph = float(channels[_CH_SPEED])
        if _CH_THROTTLE in channels:
            self.throttle = max(0.0, min(1.0, float(channels[_CH_THROTTLE]) / 100.0))
        if _CH_BRAKE in channels:
            raw = float(channels[_CH_BRAKE])
            self.brake = max(0.0, min(1.0, raw / 100.0 if raw > 1.0 else raw))
        self.car_data_samples += 1

    def update_weather(self, data: dict) -> None:
        self.track_temp_c = float(data.get("TrackTemp", self.track_temp_c))
        self.air_temp_c = float(data.get("AirTemp", self.air_temp_c))
        raw_humidity = float(data.get("Humidity", self.humidity * 100))
        self.humidity = raw_humidity / 100.0 if raw_humidity > 1.0 else raw_humidity

    def should_emit(self) -> bool:
        return self.car_data_samples > 0 and self.car_data_samples % self._emit_interval == 0

    def make_event(self, lap_time_s: float | None = None) -> TelemetryEvent:
        thermal = self.track_temp_c * 0.28 + self.brake * 7.0 + self.throttle * 2.5
        tire_t = self.tire_temp_base + thermal
        self.brake_temp = 360.0 + self.brake * 470.0 + max(0.0, self.speed_kph - 180.0) * 0.8
        self.fuel_kg = max(0.0, 105.0 - (self.lap - 1) * 1.72)
        self.ers_soc = max(0.05, min(1.0, 0.86 - self.lap * 0.012 + (3 - self.sector) * 0.018))
        self.ers_deployment_kw = max(0.0, min(120.0, self.throttle * 92.0 - self.brake * 35.0))
        return TelemetryEvent(
            session_id=self.session_id,
            car_id=self.driver,
            lap=self.lap,
            sector=self.sector,
            speed_kph=self.speed_kph,
            throttle=self.throttle,
            brake=self.brake,
            steering_angle=self.steering_angle,
            tire_temp_fl=tire_t + self.lateral_g * 1.8,
            tire_temp_fr=tire_t - self.lateral_g * 1.2,
            tire_temp_rl=tire_t - 2.4 + self.throttle * 1.8,
            tire_temp_rr=tire_t - 2.0 + self.throttle * 1.6,
            brake_temp=self.brake_temp,
            slip_angle=self.slip_angle,
            lateral_g=self.lateral_g,
            ers_soc=self.ers_soc,
            ers_deployment_kw=self.ers_deployment_kw,
            fuel_kg=self.fuel_kg,
            track_temp_c=self.track_temp_c,
            air_temp_c=self.air_temp_c,
            humidity=self.humidity,
            compound=self.compound,
            lap_time_s=lap_time_s,
        )


class FastF1LiveSource:
    """Near-real-time telemetry from the F1 live timing SignalR feed.

    Subscribes to CarData.z, TimingData, and WeatherData topics. Car data
    samples are accumulated and flushed as TelemetryEvents on a cadence
    (~2.5 seconds at default 25-sample intervals). Observed lap times from
    TimingData are included when a lap boundary is crossed.

    Requires fastf1 and signalrcore to be installed.
    An F1 account may be required for car-level channels during a session.
    Pass no_auth=True to attempt unauthenticated access (may return partial data).
    """

    def __init__(
        self,
        driver: str,
        session_id: str,
        recording_path: str = "data/live-timing-recording.txt",
        on_event: Callable[[TelemetryEvent], None] | None = None,
        no_auth: bool = False,
        timeout: int = 120,
    ) -> None:
        self._driver = driver.upper()
        self._session_id = session_id
        self._recording_path = recording_path
        self._on_event = on_event
        self._no_auth = no_auth
        self._timeout = timeout
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._driver_states: dict[str, _DriverState] = {}
        self._event_queue: queue.Queue[TelemetryEvent] = queue.Queue(maxsize=500)
        self._message_count = 0
        self._event_count = 0
        self._lock = threading.Lock()
        self._event_times: list[float] = []
        self.status = StreamStatus(
            mode="live",
            session_id=session_id,
            driver=driver,
        )

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            from fastf1.livetiming.client import SignalRClient
        except ImportError:
            with self._lock:
                self.status.error = "fastf1 not installed — pip install -e '[data]'"
            return

        Path(self._recording_path).parent.mkdir(parents=True, exist_ok=True)

        client = _CallbackSignalRClient(
            filename=self._recording_path,
            live_callback=self._on_raw_message,
            no_auth=self._no_auth,
            timeout=self._timeout,
            base_class=SignalRClient,
        )
        with self._lock:
            self.status.connected = True
        try:
            client.start()
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            with self._lock:
                self.status.error = str(exc)
        finally:
            with self._lock:
                self.status.connected = False

    def _on_raw_message(self, msg: object) -> None:
        with self._lock:
            self._message_count += 1
            self.status.message_count = self._message_count
        try:
            from signalrcore.messages.completion_message import CompletionMessage
            if isinstance(msg, CompletionMessage) and msg.result:
                for topic, data in msg.result.items():
                    self._process_topic(topic, data)
            elif isinstance(msg, list) and len(msg) >= 2:
                self._process_topic(str(msg[0]), msg[1])
        except Exception:
            pass

    def _process_topic(self, topic: str, data: object) -> None:
        if topic == "CarData.z":
            self._handle_car_data(data)
        elif topic == "TimingData":
            self._handle_timing(data)
        elif topic == "WeatherData":
            self._handle_weather(data)

    def _handle_car_data(self, data: object) -> None:
        parsed = _decompress_z(data)
        if parsed is None:
            return
        for entry in parsed.get("Entries", []):
            cars = entry.get("Cars", {})
            for driver_num, car in cars.items():
                channels = car.get("Channels", {})
                state = self._get_or_create_state(driver_num)
                state.update_car_data(channels)
                if state.should_emit():
                    event = state.make_event()
                    self._emit(event, driver_num)

    def _handle_timing(self, data: object) -> None:
        if not isinstance(data, dict):
            return
        lines = data.get("Lines", {})
        for driver_num, timing in lines.items():
            state = self._driver_states.get(driver_num)
            if state is None:
                continue
            new_lap = timing.get("NumberOfLaps")
            if new_lap is not None:
                try:
                    new_lap_int = int(new_lap)
                except (TypeError, ValueError):
                    continue
                if new_lap_int > state.lap:
                    raw_time = (timing.get("LastLapTime") or {}).get("Value", "")
                    lap_time_s = _parse_lap_time(raw_time)
                    event = state.make_event(lap_time_s=lap_time_s)
                    state.latest_lap_time_s = lap_time_s
                    self._emit(event, driver_num)
                    state.lap = new_lap_int
                    state.sector = 1
            stints = timing.get("Stints")
            if stints:
                latest_stint = list(stints.values())[-1] if isinstance(stints, dict) else stints[-1]
                compound_raw = (latest_stint or {}).get("Compound", "")
                if compound_raw:
                    state.compound = _COMPOUND_MAP.get(compound_raw.lower(), TireCompound.MEDIUM)

    def _handle_weather(self, data: object) -> None:
        if not isinstance(data, dict):
            return
        for state in self._driver_states.values():
            state.update_weather(data)

    def _get_or_create_state(self, driver_num: str) -> _DriverState:
        if driver_num not in self._driver_states:
            self._driver_states[driver_num] = _DriverState(
                session_id=self._session_id,
                driver=driver_num,
            )
        return self._driver_states[driver_num]

    def _emit(self, event: TelemetryEvent, driver_num: str) -> None:
        if self._driver and driver_num != self._driver:
            return
        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                pass
        with self._lock:
            self._event_count += 1
            self._event_times.append(time.monotonic())
            self._event_times = [t for t in self._event_times if t > time.monotonic() - 5]
            self.status.events_ingested = self._event_count
            self.status.events_per_second = len(self._event_times) / 5.0
            self.status.latest_lap = event.lap
            self.status.latest_lap_time_s = event.lap_time_s
            self.status.current_compound = event.compound.value


class _CallbackSignalRClient:
    """Wraps SignalRClient to fire a callback before writing each message."""

    def __init__(
        self,
        filename: str,
        live_callback: Callable,
        base_class: type,
        **kwargs: object,
    ) -> None:
        self._cb = live_callback
        self._inner = base_class(filename=filename, **kwargs)
        original_on_message = self._inner._on_message

        def patched_on_message(msg: object) -> None:
            self._cb(msg)
            original_on_message(msg)

        self._inner._on_message = patched_on_message  # type: ignore[method-assign]

    def start(self) -> None:
        self._inner.start()


# ── Manager ───────────────────────────────────────────────────────────────────

class LiveStreamManager:
    """Thread-safe manager for whichever live source is active.

    Call configure_replay() or configure_live() to set the source, then
    start() / stop() to control the stream. status() is always safe to call.
    The on_event callback is called from a background thread — callers must
    handle thread safety for their own state.
    """

    def __init__(self) -> None:
        self._source: ReplayStreamSource | FastF1LiveSource | None = None
        self._on_event: Callable[[TelemetryEvent], None] | None = None
        self._lock = threading.Lock()

    def set_event_callback(self, callback: Callable[[TelemetryEvent], None]) -> None:
        self._on_event = callback

    def configure_replay(
        self,
        dataset_path: str,
        speed_multiplier: float = 1.0,
    ) -> StreamStatus:
        with self._lock:
            self._stop_active()
            self._source = ReplayStreamSource(
                dataset_path=dataset_path,
                speed_multiplier=speed_multiplier,
                on_event=self._on_event,
            )
            return self._source.status

    def configure_live(
        self,
        driver: str,
        session_id: str,
        recording_path: str = "data/live-timing-recording.txt",
        no_auth: bool = False,
        timeout: int = 120,
    ) -> StreamStatus:
        with self._lock:
            self._stop_active()
            self._source = FastF1LiveSource(
                driver=driver,
                session_id=session_id,
                recording_path=recording_path,
                on_event=self._on_event,
                no_auth=no_auth,
                timeout=timeout,
            )
            return self._source.status

    def start(self) -> StreamStatus:
        with self._lock:
            if self._source is None:
                raise RuntimeError("No source configured. Call configure_replay or configure_live first.")
            self._source.start()
            return self._source.status

    def stop(self) -> None:
        with self._lock:
            self._stop_active()

    def status(self) -> StreamStatus:
        with self._lock:
            if self._source is None:
                return StreamStatus()
            return self._source.status

    def _stop_active(self) -> None:
        if self._source is not None:
            self._source.stop()
            self._source = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decompress_z(data: object) -> dict | None:
    if isinstance(data, dict):
        return data
    if not isinstance(data, str):
        return None
    try:
        return json.loads(zlib.decompress(base64.b64decode(data)))
    except Exception:
        return None


def _parse_lap_time(value: str | None) -> float | None:
    if not value:
        return None
    value = str(value).strip()
    try:
        if ":" in value:
            parts = value.split(":")
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds
        return float(value)
    except (ValueError, IndexError):
        return None


def _row_to_event(row: dict) -> TelemetryEvent | None:
    try:
        compound_raw = str(row.get("compound") or "medium").strip().lower()
        compound = _COMPOUND_MAP.get(compound_raw, TireCompound.MEDIUM)
        lap_time_raw = row.get("lap_time_s")
        lap_time_s: float | None = None
        if lap_time_raw not in (None, "", "None", "null"):
            lap_time_s = float(lap_time_raw)  # type: ignore[arg-type]
        return TelemetryEvent(
            session_id=str(row.get("session_id") or "live-replay"),
            car_id=str(row.get("car_id") or "unknown"),
            lap=int(float(row.get("lap") or 1)),
            sector=int(float(row.get("sector") or 1)),
            speed_kph=float(row.get("speed_kph") or 200.0),
            throttle=float(row.get("throttle") or 0.7),
            brake=float(row.get("brake") or 0.0),
            steering_angle=float(row.get("steering_angle") or 0.0),
            tire_temp_fl=float(row.get("tire_temp_fl") or 88.0),
            tire_temp_fr=float(row.get("tire_temp_fr") or 88.0),
            tire_temp_rl=float(row.get("tire_temp_rl") or 86.0),
            tire_temp_rr=float(row.get("tire_temp_rr") or 86.0),
            brake_temp=float(row.get("brake_temp") or 420.0),
            slip_angle=float(row.get("slip_angle") or 2.0),
            lateral_g=float(row.get("lateral_g") or 1.8),
            ers_soc=float(row.get("ers_soc") or 0.75),
            ers_deployment_kw=float(row.get("ers_deployment_kw") or 60.0),
            fuel_kg=float(row.get("fuel_kg") or 80.0),
            track_temp_c=float(row.get("track_temp_c") or 37.0),
            air_temp_c=float(row.get("air_temp_c") or 27.0),
            humidity=float(row.get("humidity") or 0.45),
            compound=compound,
            lap_time_s=lap_time_s,
            timestamp_ms=int(float(row.get("timestamp_ms") or 0)),
        )
    except Exception:
        return None
