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
class RecoveryRule:
    condition: str
    strategy: str
    candidate_skills: tuple = ()
    candidate_categories: tuple = ()
    guidance: tuple = ()
    return_to_skill: Optional[str] = None

    def as_dict(self):
        result = {"condition": self.condition, "strategy": self.strategy}
        for key, value in (
            ("candidate_skills", self.candidate_skills),
            ("candidate_categories", self.candidate_categories),
            ("guidance", self.guidance),
        ):
            if value:
                result[key] = list(value)
        if self.return_to_skill:
            result["return_to_skill"] = self.return_to_skill
        return result


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    instruction: str
    planner_instruction: str
    oracle_task: str
    policy_group: str
    category: str
    grounding_roles: tuple
    max_steps: int
    max_attempts: int
    replan_interval: int
    success_guard: Optional[Dict[str, Any]] = None
    planner_guidance: Optional[PlannerGuidance] = None
    recovery: tuple = ()
