from __future__ import annotations

from collections import defaultdict, deque
from math import log2
from statistics import mean, pstdev

from f1_strategy.domain import OnlineFeatures, TelemetryEvent


class OnlineFeatureStore:
    """In-memory online feature store for low-latency local inference.

    The interface is deliberately small so this can later be backed by Redis,
    Feast, ClickHouse, or a stream processor without changing inference code.
    """

    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self._events: dict[tuple[str, str], deque[TelemetryEvent]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._features: dict[tuple[str, str], OnlineFeatures] = {}
        self._stint_start_lap: dict[tuple[str, str], int] = {}

    def ingest(self, event: TelemetryEvent) -> OnlineFeatures:
        key = (event.session_id, event.car_id)
        events = self._events[key]
        if not events or events[-1].compound != event.compound:
            self._stint_start_lap[key] = event.lap
        events.append(event)
        features = self._build_features(key, events)
        self._features[key] = features
        return features

    def get(self, session_id: str, car_id: str) -> OnlineFeatures | None:
        return self._features.get((session_id, car_id))

    def snapshot(self) -> list[OnlineFeatures]:
        return list(self._features.values())

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

        return OnlineFeatures(
            session_id=latest.session_id,
            car_id=latest.car_id,
            lap=latest.lap,
            compound=latest.compound,
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
            rolling_lap_time_s=mean(rolling_laps[-5:]) if rolling_laps else 90.0,
            dirty_air_risk=self._dirty_air_risk(events),
        )

    @staticmethod
    def _wear_proxy(events: list[TelemetryEvent]) -> float:
        if not events:
            return 0.0
        return mean(OnlineFeatureStore._event_wear_proxy(e) for e in events)

    @staticmethod
    def _event_wear_proxy(event: TelemetryEvent) -> float:
        return (
            event.lateral_g * 0.38
            + abs(event.slip_angle) * 0.05
            + max(event.track_temp_c - 35.0, 0.0) * 0.015
            + event.brake * 0.20
        )

    @staticmethod
    def _dirty_air_risk(events: deque[TelemetryEvent]) -> float:
        if len(events) < 3:
            return 0.0
        throttle = mean(e.throttle for e in events)
        steering_std = pstdev(e.steering_angle for e in events)
        high_speed_understeer = max(0.0, steering_std / 20.0 - throttle)
        return min(1.0, high_speed_understeer)

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
