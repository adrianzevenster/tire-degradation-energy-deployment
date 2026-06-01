from __future__ import annotations

from f1_strategy.domain import EnergyPlan, OnlineFeatures, PitWindow, Prediction, StrategyRecommendation
from f1_strategy.models import COMPOUND_LIFE, ModelConfig


class StrategyOptimizer:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()

    def recommend(
        self, features: OnlineFeatures, prediction: Prediction, remaining_laps: int
    ) -> StrategyRecommendation:
        pit_window = self._pit_window(features, prediction, remaining_laps)
        energy_plan = self._energy_plan(features)
        reasons = self._reasons(features, prediction, pit_window)
        pace_target = min(0.85, max(-0.35, prediction.next_lap_delta_s - 0.18))
        return StrategyRecommendation(
            session_id=features.session_id,
            car_id=features.car_id,
            prediction=prediction,
            pit_window=pit_window,
            energy_plan=energy_plan,
            pace_target_delta_s=pace_target,
            reasons=reasons,
        )

    def _pit_window(
        self, features: OnlineFeatures, prediction: Prediction, remaining_laps: int
    ) -> PitWindow:
        natural_life = COMPOUND_LIFE[features.compound]
        predicted_cliff_lap = features.lap + int(max(1.0, prediction.remaining_tire_life_laps))
        crossover = features.lap + max(1, int(remaining_laps * 0.45))
        target = min(predicted_cliff_lap - 2, crossover)
        target = max(features.lap + 1, target)
        earliest = max(features.lap + 1, target - 3)
        latest = min(features.lap + remaining_laps, target + 4, int(features.lap + natural_life))
        undercut = min(
            0.92,
            max(0.08, 0.38 + prediction.grip_loss_pct / 80.0 + prediction.cliff_probability * 0.25),
        )
        safety_car = min(1.0, 0.18 + prediction.cliff_probability * 0.35 + remaining_laps / 160.0)
        return PitWindow(
            earliest_lap=earliest,
            target_lap=target,
            latest_lap=max(latest, earliest),
            undercut_success_probability=undercut,
            safety_car_sensitivity=safety_car,
        )

    @staticmethod
    def _energy_plan(features: OnlineFeatures) -> EnergyPlan:
        usable_kw = min(120.0, max(0.0, features.ers_soc * 160.0))
        if features.ers_soc < 0.35:
            deployment = {1: usable_kw * 0.20, 2: usable_kw * 0.15, 3: usable_kw * 0.10}
        elif features.dirty_air_risk > 0.45:
            deployment = {1: usable_kw * 0.35, 2: usable_kw * 0.25, 3: usable_kw * 0.30}
        else:
            deployment = {1: usable_kw * 0.25, 2: usable_kw * 0.30, 3: usable_kw * 0.25}
        total_deployment = sum(deployment.values())
        ending_soc = max(0.05, features.ers_soc - total_deployment / 520.0 + 0.08)
        lap_gain = min(0.65, total_deployment / 260.0 * max(0.55, features.ers_efficiency / 1.4))
        return EnergyPlan(
            sector_deployment_kw={sector: round(value, 1) for sector, value in deployment.items()},
            expected_lap_gain_s=round(lap_gain, 3),
            ending_soc=round(ending_soc, 3),
        )

    @staticmethod
    def _reasons(
        features: OnlineFeatures, prediction: Prediction, pit_window: PitWindow
    ) -> list[str]:
        reasons = []
        if prediction.cliff_probability > 0.55:
            reasons.append("High tire-cliff probability; protect target pit window.")
        if prediction.overheating_probability > 0.60:
            reasons.append("Thermal degradation is elevated; reduce sliding and brake migration.")
        if features.ers_soc < 0.35:
            reasons.append("ERS state of charge is low; bias toward recharge.")
        if pit_window.undercut_success_probability > 0.65:
            reasons.append("Undercut probability is favorable based on degradation-adjusted pace.")
        if not reasons:
            reasons.append("Current stint is stable; maintain pace target and monitor drift.")
        return reasons
