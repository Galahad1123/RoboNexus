"""Legacy backend-neutral retry state machine.

Environment-specific executors own their rollout loop. This controller only
decides whether recovery and another independently initialized attempt occur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .interfaces import EventSink
from .schemas import AttemptResult, MotionResult, MotionStatus, TaskSpec


@dataclass(frozen=True)
class RetryPolicy:
    enabled: bool = True
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")


class TaskRetryController:
    """Coordinate attempts while preserving the backend's current scene."""

    def __init__(self, events: EventSink) -> None:
        self.events = events

    def run(
        self,
        task: TaskSpec,
        policy: RetryPolicy,
        execute_attempt: Callable[[int], AttemptResult],
        recover: Callable[[int, Any], MotionResult],
    ) -> list[AttemptResult]:
        limit = policy.max_attempts if policy.enabled else 1
        attempts: list[AttemptResult] = []
        observation: Any = None
        for attempt in range(1, limit + 1):
            if attempt > 1:
                self.events.emit(
                    "recovery_started",
                    task=task.task_id,
                    attempt=attempt,
                )
                motion = recover(attempt, observation)
                self.events.emit(
                    "recovery_completed"
                    if motion.status == MotionStatus.REACHED
                    else "recovery_failed",
                    task=task.task_id,
                    attempt=attempt,
                    status=motion.status.value,
                    steps=motion.steps,
                    reason=motion.reason,
                )
                if motion.status != MotionStatus.REACHED:
                    break
            self.events.emit("attempt_started", task=task.task_id, attempt=attempt)
            result = execute_attempt(attempt)
            attempts.append(result)
            observation = result.observation
            self.events.emit(
                "attempt_succeeded" if result.success else "attempt_failed",
                task=task.task_id,
                attempt=attempt,
                steps=result.steps,
                reason=result.termination_reason,
            )
            if result.success:
                break
        return attempts
