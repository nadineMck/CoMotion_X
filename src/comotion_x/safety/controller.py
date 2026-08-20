"""Unaware, reactive, and predictive velocity-scaling safety controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from comotion_x.core.models import ControlCommand, SafetyMode
from comotion_x.safety.risk import CollisionRisk

MODE_RANK = {
    SafetyMode.NORMAL: 0,
    SafetyMode.CAUTION: 1,
    SafetyMode.HIGH_RISK: 2,
    SafetyMode.CRITICAL: 3,
}


@dataclass(frozen=True, slots=True)
class ControllerParameters:
    caution_clearance_m: float = 0.30
    high_risk_clearance_m: float = 0.12
    critical_clearance_m: float = 0.0
    caution_velocity_scale: float = 0.60
    high_risk_velocity_scale: float = 0.25
    hysteresis_m: float = 0.03
    minimum_dwell_seconds: float = 0.20

    def __post_init__(self) -> None:
        if not (
            self.critical_clearance_m
            < self.high_risk_clearance_m
            < self.caution_clearance_m
        ):
            raise ValueError("clearance thresholds must be strictly increasing")
        if not 0 <= self.high_risk_velocity_scale < self.caution_velocity_scale <= 1:
            raise ValueError("velocity scales must be ordered within [0, 1]")


@dataclass(frozen=True, slots=True)
class ControllerDecision:
    timestamp: float
    mode: SafetyMode
    command: ControlCommand
    evaluated_clearance_m: float | None
    reason: str


class SafetyController(Protocol):
    def update(
        self,
        timestamp: float,
        *,
        current_clearance_m: float,
        predicted_risk: CollisionRisk | None,
    ) -> ControllerDecision: ...


class NoAwarenessController:
    def update(
        self,
        timestamp: float,
        *,
        current_clearance_m: float,
        predicted_risk: CollisionRisk | None,
    ) -> ControllerDecision:
        return ControllerDecision(
            timestamp=timestamp,
            mode=SafetyMode.NORMAL,
            command=ControlCommand(velocity_scale=1.0),
            evaluated_clearance_m=None,
            reason="human awareness disabled",
        )


class ClearanceController:
    """Mode state machine shared by reactive and predictive policies."""

    def __init__(self, parameters: ControllerParameters, *, predictive: bool) -> None:
        self.parameters = parameters
        self.predictive = predictive
        self.mode = SafetyMode.NORMAL
        self._last_transition_timestamp = 0.0

    def update(
        self,
        timestamp: float,
        *,
        current_clearance_m: float,
        predicted_risk: CollisionRisk | None,
    ) -> ControllerDecision:
        if timestamp < 0:
            raise ValueError("controller timestamp must be non-negative")
        if self.predictive:
            if predicted_risk is None:
                clearance = current_clearance_m
                reason = "current clearance during prediction warm-up"
            else:
                clearance = predicted_risk.minimum_clearance_m
                reason = "minimum predicted clearance"
        else:
            clearance = current_clearance_m
            reason = "current measured clearance"

        requested_mode = self._classify(clearance)
        next_mode = self._transition(timestamp, clearance, requested_mode)
        if next_mode is not self.mode:
            self.mode = next_mode
            self._last_transition_timestamp = timestamp
        return ControllerDecision(
            timestamp=timestamp,
            mode=self.mode,
            command=self._command(self.mode),
            evaluated_clearance_m=clearance,
            reason=reason,
        )

    def _classify(self, clearance_m: float) -> SafetyMode:
        if clearance_m <= self.parameters.critical_clearance_m:
            return SafetyMode.CRITICAL
        if clearance_m < self.parameters.high_risk_clearance_m:
            return SafetyMode.HIGH_RISK
        if clearance_m < self.parameters.caution_clearance_m:
            return SafetyMode.CAUTION
        return SafetyMode.NORMAL

    def _transition(
        self, timestamp: float, clearance_m: float, requested_mode: SafetyMode
    ) -> SafetyMode:
        if MODE_RANK[requested_mode] > MODE_RANK[self.mode]:
            return requested_mode
        if MODE_RANK[requested_mode] == MODE_RANK[self.mode]:
            return self.mode
        if timestamp - self._last_transition_timestamp < self.parameters.minimum_dwell_seconds:
            return self.mode

        release_threshold = {
            SafetyMode.CRITICAL: self.parameters.critical_clearance_m,
            SafetyMode.HIGH_RISK: self.parameters.high_risk_clearance_m,
            SafetyMode.CAUTION: self.parameters.caution_clearance_m,
        }.get(self.mode)
        if release_threshold is None:
            return SafetyMode.NORMAL
        if clearance_m <= release_threshold + self.parameters.hysteresis_m:
            return self.mode
        next_rank = MODE_RANK[self.mode] - 1
        return next(mode for mode, rank in MODE_RANK.items() if rank == next_rank)

    def _command(self, mode: SafetyMode) -> ControlCommand:
        if mode is SafetyMode.CRITICAL:
            return ControlCommand(velocity_scale=0.0, stop=True)
        if mode is SafetyMode.HIGH_RISK:
            return ControlCommand(velocity_scale=self.parameters.high_risk_velocity_scale)
        if mode is SafetyMode.CAUTION:
            return ControlCommand(velocity_scale=self.parameters.caution_velocity_scale)
        return ControlCommand(velocity_scale=1.0)


class ReactiveSafetyController(ClearanceController):
    def __init__(self, parameters: ControllerParameters) -> None:
        super().__init__(parameters, predictive=False)


class PredictiveSafetyController(ClearanceController):
    def __init__(self, parameters: ControllerParameters) -> None:
        super().__init__(parameters, predictive=True)
