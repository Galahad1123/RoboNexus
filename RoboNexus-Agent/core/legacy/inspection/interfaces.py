"""Legacy state inspection interface retained for scripted regression."""

from __future__ import annotations

from typing import Any, Protocol

from .schemas import SkillInspection


class StateInspector(Protocol):
    def inspect(self, skill_id: str, observation: Any) -> SkillInspection: ...
