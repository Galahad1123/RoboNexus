"""Domain types for canonical atomic skills."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SkillDefaults:
    max_steps: int
    max_attempts: int
    replan_interval: int


@dataclass(frozen=True)
class PlannerGuidance:
    """Declarative, backend-neutral hints exposed to the high-level planner."""

    planning: tuple = ()
    grounding: tuple = ()
    recovery: tuple = ()

    def as_dict(self):
        return {
            key: list(value)
            for key, value in (
                ("planning", self.planning),
                ("grounding", self.grounding),
                ("recovery", self.recovery),
            )
            if value
        }


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    instruction: str
    oracle_task: str
    policy_group: str
    grounding_roles: tuple
    max_steps: int
    max_attempts: int
    replan_interval: int
    success_guard: Optional[Dict[str, Any]] = None
    planner_guidance: Optional[PlannerGuidance] = None
