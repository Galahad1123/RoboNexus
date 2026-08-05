"""Legacy retry-era execution interfaces retained for regression."""

from __future__ import annotations

from typing import Any, Protocol

from .schemas import (
    ApproachPlan,
    AttemptResult,
    LocalizationRequest,
    MotionResult,
    TargetEstimate,
    TaskSpec,
    VerificationResult,
)


class TaskVerifier(Protocol):
    def begin(self, task: TaskSpec, initial_observation: Any) -> None: ...
    def evaluate(self, observation: Any, context: dict[str, Any]) -> VerificationResult: ...


class TargetLocalizer(Protocol):
    def locate(
        self, request: LocalizationRequest, observation: Any
    ) -> TargetEstimate: ...


class ApproachPoseProvider(Protocol):
    def compute(
        self, task: TaskSpec, target: TargetEstimate, robot_state: Any
    ) -> ApproachPlan: ...


class RobotMotionController(Protocol):
    def execute(self, plan: ApproachPlan) -> MotionResult: ...


class PolicyExecutor(Protocol):
    def execute_attempt(self, task: TaskSpec, attempt: int) -> AttemptResult: ...


class EventSink(Protocol):
    def emit(self, event: str, **fields: Any) -> None: ...
