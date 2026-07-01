from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import asdict
from math import log2
from pathlib import Path
from statistics import mean
from typing import Any, Protocol, runtime_checkable

from f1_strategy.domain import FleetState, OnlineFeatures, TelemetryEvent, TireCompound


def _pstdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5


@runtime_checkable
class FeatureStore(Protocol):
    """Minimal interface shared by all feature store backends."""

    def ingest(self, event: TelemetryEvent) -> OnlineFeatures: ...
    def get(self, session_id: str, car_id: str) -> OnlineFeatures | None: ...
    def snapshot(self) -> list[OnlineFeatures]: ...


# Compound-specific mechanical wear multipliers relative to medium baseline.
# Soft wears ~45% faster; hard ~30% slower. Wet/inter primarily thermal, not mechanical.
_COMPOUND_WEAR_MULTIPLIER: dict[TireCompound, float] = {
    TireCompound.SOFT: 1.45,
    TireCompound.MEDIUM: 1.00,
    TireCompound.HARD: 0.70,
    TireCompound.INTERMEDIATE: 0.85,
    TireCompound.WET: 0.60,
}


class OnlineFeatureStore:
    """In-memory online feature store for low-latency local inference.

    Supports optional fleet state integration (set_fleet_state) for gap-aware
    dirty_air_risk computation, and checkpoint/restore for cross-restart durability.
    """

    def __init__(self, window_size: int = 50, base_lap_time_s: float = 90.0) -> None:
        self.window_size = window_size
        self._base_lap_time_s = base_lap_time_s
        self._events: dict[tuple[str, str], deque[TelemetryEvent]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._features: dict[tuple[str, str], OnlineFeatures] = {}
        self._stint_start_lap: dict[tuple[str, str], int] = {}
        self._stint_entry_lt: dict[tuple[str, str], float] = {}
        self._prev_lap_time: dict[tuple[str, str], float] = {}
        self._fleet_state: FleetState | None = None

    def set_fleet_state(self, fleet_state: FleetState) -> None:
        """Attach live fleet gap data; improves dirty_air_risk and optimizer inputs."""
        self._fleet_state = fleet_state

    def ingest(self, event: TelemetryEvent) -> OnlineFeatures:
        key = (event.session_id, event.car_id)
        events = self._events[key]
        if not events or events[-1].compound != event.compound:
            self._stint_start_lap[key] = event.lap
            self._stint_entry_lt.pop(key, None)
            self._prev_lap_time.pop(key, None)
        if key not in self._stint_entry_lt and event.lap_time_s is not None:
            self._stint_entry_lt[key] = event.lap_time_s
        events.append(event)
        features = self._build_features(key, events)
        self._features[key] = features
        return features

    def get(self, session_id: str, car_id: str) -> OnlineFeatures | None:
        return self._features.get((session_id, car_id))

    def snapshot(self) -> list[OnlineFeatures]:
        return list(self._features.values())

    def checkpoint(self, path: Path | str) -> None:
        """Serialize event windows and stint state to a JSON file for cross-restart durability."""
        data: dict[str, Any] = {
            "window_size": self.window_size,
            "base_lap_time_s": self._base_lap_time_s,
            "events": {
                f"{sid}:{cid}": [
                    {**asdict(e), "compound": e.compound.value}
                    for e in evts
                ]
                for (sid, cid), evts in self._events.items()
            },
            "stint_start_lap": {
                f"{sid}:{cid}": v for (sid, cid), v in self._stint_start_lap.items()
            },
            "stint_entry_lt": {
                f"{sid}:{cid}": v for (sid, cid), v in self._stint_entry_lt.items()
            },
            "prev_lap_time": {
                f"{sid}:{cid}": v for (sid, cid), v in self._prev_lap_time.items()
            },
        }
        Path(path).write_text(json.dumps(data, default=str), encoding="utf-8")

    @classmethod
    def from_checkpoint(cls, path: Path | str) -> "OnlineFeatureStore":
        """Restore a feature store from a JSON checkpoint file."""
        from f1_strategy.serialization import telemetry_from_dict

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(
            window_size=data["window_size"],
            base_lap_time_s=data["base_lap_time_s"],
        )
        for key_str, raw_events in data["events"].items():
            sid, cid = key_str.split(":", 1)
            for raw in raw_events:
                raw["compound"] = TireCompound(raw["compound"])
                store.ingest(telemetry_from_dict({**raw, "session_id": sid, "car_id": cid}))
        return store

    def _build_features(
        self, key: tuple[str, str], events: deque[TelemetryEvent]
    ) -> OnlineFeatures:
        latest = events[-1]
        tire_temps = [
            latest.tire_temp_fl,
            latest.tire_temp_fr,
            latest.tire_temp_rl,
            latest.tire_temp_rr,
        ]
        rolling_laps = [e.lap_time_s for e in events if e.lap_time_s is not None]
        rolling_lt = mean(rolling_laps[-5:]) if rolling_laps else self._base_lap_time_s
        entry_lt = self._stint_entry_lt.get(key)
        stint_lap_delta_s = rolling_lt - entry_lt if entry_lt is not None else 0.0
        prev_lt = self._prev_lap_time.get(key)
        lap_to_lap_delta_s = rolling_lt - prev_lt if prev_lt is not None else 0.0
        if latest.lap_time_s is not None:
            self._prev_lap_time[key] = latest.lap_time_s
        brake_heat_index = mean(e.brake * e.brake_temp for e in events)
        aggression = mean(
            0.40 * abs(e.steering_angle) / 45.0
            + 0.35 * e.brake
            + 0.25 * abs(e.slip_angle) / 12.0
            for e in events
        )
        ers_used = sum(max(e.ers_deployment_kw, 0.0) for e in events)
        speed_gain = max(mean(e.speed_kph for e in events) - 180.0, 1.0)
        previous_wear_proxy = self._wear_proxy(list(events)[:-8])
        current_wear_proxy = self._wear_proxy(list(events)[-8:])

        car_id = latest.car_id
        fleet = self._fleet_state
        fleet_feats = fleet.as_features(car_id) if fleet is not None else {}

        return OnlineFeatures(
            session_id=latest.session_id,
            car_id=car_id,
            lap=latest.lap,
            compound=latest.compound,
            circuit=latest.circuit,
            tire_age_laps=max(0, latest.lap - self._stint_start_lap.get(key, latest.lap)),
            mean_tire_temp=mean(tire_temps),
            tire_temp_gradient=max(tire_temps) - min(tire_temps),
            brake_heat_index=brake_heat_index,
            driver_aggression=min(1.5, aggression),
            steering_entropy=self._entropy([e.steering_angle for e in events]),
            cumulative_tire_load=sum(self._event_wear_proxy(e) for e in events)
            / max(1, latest.sector),
            degradation_acceleration=max(0.0, current_wear_proxy - previous_wear_proxy),
            ers_efficiency=min(1.4, speed_gain / max(ers_used / max(len(events), 1), 1.0)),
            ers_soc=latest.ers_soc,
            track_temp_c=latest.track_temp_c,
            fuel_kg=latest.fuel_kg,
            rolling_lap_time_s=rolling_lt,
            dirty_air_risk=self._dirty_air_risk(events, fleet_gap_ahead_s=fleet_feats.get("fleet_gap_ahead_s", 999.0)),  # type: ignore[arg-type]
            circuit_base_lap_s=self._base_lap_time_s,
            stint_lap_delta_s=stint_lap_delta_s,
            lap_to_lap_delta_s=lap_to_lap_delta_s,
            fleet_gap_ahead_s=float(fleet_feats.get("fleet_gap_ahead_s", 999.0)),
            fleet_gap_behind_s=float(fleet_feats.get("fleet_gap_behind_s", 999.0)),
            fleet_position=int(fleet_feats.get("fleet_position", 10)),
            fleet_competitor_tire_age=int(fleet_feats.get("fleet_competitor_tire_age", 0)),
            fleet_competitor_compound=fleet_feats.get("fleet_competitor_compound", TireCompound.MEDIUM),  # type: ignore[arg-type]
        )

    @staticmethod
    def _wear_proxy(events: list[TelemetryEvent]) -> float:
        if not events:
            return 0.0
        return mean(OnlineFeatureStore._event_wear_proxy(e) for e in events)

    @staticmethod
    def _event_wear_proxy(event: TelemetryEvent) -> float:
        # Compound-specific mechanical degradation: softs degrade faster under the same loads.
        base = (
            event.lateral_g * 0.38
            + abs(event.slip_angle) * 0.05
            + max(event.track_temp_c - 35.0, 0.0) * 0.015
            + event.brake * 0.20
        )
        return base * _COMPOUND_WEAR_MULTIPLIER.get(event.compound, 1.0)

    @staticmethod
    def _dirty_air_risk(events: deque[TelemetryEvent], fleet_gap_ahead_s: float = 999.0) -> float:
        if len(events) < 3:
            return 0.0
        throttle = mean(e.throttle for e in events)
        steering_std = _pstdev([e.steering_angle for e in events])
        telemetry_risk = max(0.0, steering_std / 20.0 - throttle)

        # If we have real gap data, gate the telemetry proxy: dirty air is only relevant
        # when running within ~1.5s of the car ahead.
        if fleet_gap_ahead_s < 999.0:
            gap_factor = max(0.0, 1.0 - fleet_gap_ahead_s / 1.5)
            return min(1.0, max(telemetry_risk * 0.4, gap_factor))

        return min(1.0, telemetry_risk)

    @staticmethod
    def _entropy(values: list[float], bins: int = 8) -> float:
        if not values:
            return 0.0
        lo = min(values)
        hi = max(values)
        if hi == lo:
            return 0.0
        counts = [0] * bins
        for value in values:
            index = min(bins - 1, int((value - lo) / (hi - lo) * bins))
            counts[index] += 1
        total = len(values)
        return -sum((count / total) * log2(count / total) for count in counts if count)


class RedisFeatureStore:
    """Write-through Redis-backed feature store for cross-restart durability.

    Maintains a local in-memory cache (OnlineFeatureStore) for low-latency
    serving. On every ingest, the TelemetryEvent is also pushed to Redis so
    session state survives API restarts and can be shared across instances.

    On a cache miss (e.g. after restart), events are loaded from Redis and
    replayed through the local store before returning the feature snapshot.

    Degrades gracefully: if redis-py is not installed or the connection fails,
    the store continues as a pure in-memory store.
    """

    backend_name = "redis"
    _TTL_SECONDS = 43_200  # 12 hours — covers a full race day

    def __init__(self, redis_url: str, window_size: int = 50, base_lap_time_s: float = 90.0) -> None:
        self._url = redis_url
        self._window_size = window_size
        self._local = OnlineFeatureStore(window_size=window_size, base_lap_time_s=base_lap_time_s)
        self._redis: Any = None
        self._available = False
        self._connect()

    def _connect(self) -> None:
        try:
            import redis  # type: ignore[import-untyped]
            client = redis.from_url(self._url, decode_responses=True, socket_timeout=1.0)
            client.ping()
            self._redis = client
            self._available = True
        except Exception:
            self._available = False

    @property
    def is_connected(self) -> bool:
        return self._available

    def ingest(self, event: TelemetryEvent) -> OnlineFeatures:
        features = self._local.ingest(event)
        self._push_to_redis(event)
        return features

    def get(self, session_id: str, car_id: str) -> OnlineFeatures | None:
        cached = self._local.get(session_id, car_id)
        if cached is not None:
            return cached
        events = self._load_from_redis(session_id, car_id)
        result: OnlineFeatures | None = None
        for ev in events:
            result = self._local.ingest(ev)
        return result

    def snapshot(self) -> list[OnlineFeatures]:
        return self._local.snapshot()

    def _push_to_redis(self, event: TelemetryEvent) -> None:
        if not self._available or self._redis is None:
            return
        key = f"f1:events:{event.session_id}:{event.car_id}"
        try:
            raw = asdict(event)
            raw["compound"] = event.compound.value
            pipe = self._redis.pipeline()  # type: ignore[union-attr]
            pipe.lpush(key, json.dumps(raw))
            pipe.ltrim(key, 0, self._window_size - 1)
            pipe.expire(key, self._TTL_SECONDS)
            pipe.execute()
        except Exception:
            self._available = False

    def _load_from_redis(self, session_id: str, car_id: str) -> list[TelemetryEvent]:
        if not self._available or self._redis is None:
            return []
        key = f"f1:events:{session_id}:{car_id}"
        try:
            raw_list = self._redis.lrange(key, 0, self._window_size - 1)  # type: ignore[union-attr]
            events: list[TelemetryEvent] = []
            for raw in reversed(raw_list):
                data = json.loads(raw)
                data["compound"] = TireCompound(data["compound"])
                events.append(TelemetryEvent(**data))
            return events
        except Exception:
            return []


def create_feature_store(
    backend: str = "auto",
    redis_url: str = "",
    window_size: int = 50,
    base_lap_time_s: float = 90.0,
) -> OnlineFeatureStore | RedisFeatureStore:
    """Return a RedisFeatureStore when Redis is reachable, else OnlineFeatureStore."""
    if backend in ("redis", "auto") and redis_url:
        try:
            store = RedisFeatureStore(redis_url=redis_url, window_size=window_size, base_lap_time_s=base_lap_time_s)
            if store.is_connected:
                return store
        except Exception:
            pass
    return OnlineFeatureStore(window_size=window_size, base_lap_time_s=base_lap_time_s)
