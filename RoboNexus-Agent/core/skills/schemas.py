"""Domain types for canonical atomic skills."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SkillDefaults:
    max_steps: int
    max_attempts: int
    replan_interval: int


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
