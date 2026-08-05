"""Whole-goal completion verification boundary."""

from abc import ABC, abstractmethod


class GoalVerifierPort(ABC):
    @abstractmethod
    def verify(self, goal, observation, use_goal_reference=True):
        """Return success, failure or uncertain with evidence."""

