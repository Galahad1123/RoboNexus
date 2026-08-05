"""Goal-level execution state and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.legacy.execution.schemas import AttemptResult
from core.legacy.planning import GoalSpec


class GoalStatus(str, Enum):
    SUCCEEDED = "succeeded"
    ABORTED = "aborted"


@dataclass(frozen=True)
class GoalExecutionConfig:
    max_attempts_per_skill: int = 3
    max_replans: int = 2
    base_seed: int = 42

    def __post_init__(self) -> None:
        if self.max_attempts_per_skill < 1:
            raise ValueError("max_attempts_per_skill must be >= 1")
        if self.max_replans < 0:
            raise ValueError("max_replans must be >= 0")
        if self.base_seed < 0:
            raise ValueError("base_seed must be >= 0")


@dataclass(frozen=True)
class AtomicAttemptRequest:
    goal: GoalSpec
    skill_id: str
    attempt: int
    plan_step: int
    replans_used: int
    episode_seed: int
    observation: Any = None


@dataclass
class SkillExecutionRecord:
    skill_id: str
    plan_step: int
    replans_used: int
    attempts: list[AttemptResult] = field(default_factory=list)
    skipped: bool = False
    inspection_reason: str | None = None

    @property
    def success(self) -> bool:
        return self.skipped or bool(self.attempts and self.attempts[-1].success)


@dataclass(frozen=True)
class GoalExecutionResult:
    goal: GoalSpec
    status: GoalStatus
    completed_skill_ids: tuple[str, ...]
    skill_records: tuple[SkillExecutionRecord, ...]
    attempts_executed: int
    replans_used: int
    termination_reason: str
    observation: Any = None

    @property
    def success(self) -> bool:
        return self.status == GoalStatus.SUCCEEDED
