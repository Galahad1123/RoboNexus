"""One bounded atomic skill attempt boundary."""

from abc import ABC, abstractmethod


class SkillExecutorPort(ABC):
    @abstractmethod
    def execute(self, skill_id, groundings, attempt, logical_subtask_id):
        """Execute once in the current scene and return a structured result."""

