"""Legacy goal planning contracts retained for regression tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PlannerDecisionType(str, Enum):
    EXECUTE = "execute"
    COMPLETE = "complete"
    ABORT = "abort"


@dataclass(frozen=True)
class GoalSpec:
    goal_id: str
    instruction: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannerFeedback:
    failed_skill_id: str
    attempts: int
    termination_reason: str


@dataclass(frozen=True)
class PlannerContext:
    goal: GoalSpec
    completed_skill_ids: tuple[str, ...] = ()
    observation: Any = None
    replans_used: int = 0
    feedback: PlannerFeedback | None = None


@dataclass(frozen=True)
class PlannerDecision:
    decision_type: PlannerDecisionType
    skill_id: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.decision_type == PlannerDecisionType.EXECUTE:
            if not isinstance(self.skill_id, str) or not self.skill_id.strip():
                raise ValueError("execute decision requires one non-empty skill_id")
        elif self.skill_id is not None:
            raise ValueError("complete/abort decision cannot contain skill_id")

    @classmethod
    def execute(cls, skill_id: str, reason: str = "scripted_step") -> PlannerDecision:
        return cls(PlannerDecisionType.EXECUTE, skill_id=skill_id, reason=reason)

    @classmethod
    def complete(cls, reason: str = "plan_exhausted") -> PlannerDecision:
        return cls(PlannerDecisionType.COMPLETE, reason=reason)

    @classmethod
    def abort(cls, reason: str) -> PlannerDecision:
        return cls(PlannerDecisionType.ABORT, reason=reason)
