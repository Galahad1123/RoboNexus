"""Semantic/bbox/point to grounded-object boundary."""

from abc import ABC, abstractmethod


class GrounderPort(ABC):
    @abstractmethod
    def capabilities(self):
        """Return supported representations and semantic vocabulary."""

    @abstractmethod
    def ground(self, skill_id, observation, proposals):
        """Return backend-neutral GroundedObject values and preview artifacts."""

