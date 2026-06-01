"""Real-time F1 tire, energy, and strategy forecasting system."""

from f1_strategy.domain import Prediction, StrategyRecommendation, TelemetryEvent
from f1_strategy.engine import InferenceEngine
from f1_strategy.metadata import APP_VERSION

__version__ = APP_VERSION

__all__ = [
    "InferenceEngine",
    "Prediction",
    "StrategyRecommendation",
    "TelemetryEvent",
    "__version__",
]
