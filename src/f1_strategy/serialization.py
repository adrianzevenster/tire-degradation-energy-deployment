from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from f1_strategy.domain import JsonDict, TelemetryEvent, TireCompound


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {to_jsonable(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def telemetry_from_dict(payload: JsonDict) -> TelemetryEvent:
    payload = dict(payload)
    payload["compound"] = TireCompound(payload["compound"])
    return TelemetryEvent(**payload)
