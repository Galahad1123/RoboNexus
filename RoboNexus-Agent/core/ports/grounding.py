"""Semantic/bbox/point to grounded-object boundary."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class GrounderPortError(RuntimeError):
    pass


class UnsupportedGroundingError(GrounderPortError):
    pass


class GroundingNotFoundError(GrounderPortError):
    pass


@dataclass(frozen=True)
class GroundingBatch:
    groundings: tuple
    preview: object
    artifacts: tuple = ()


class GrounderPort(ABC):
    @abstractmethod
    def capabilities(self):
        """Return supported representations and semantic vocabulary."""

    @abstractmethod
    def ground(self, skill_id, observation, proposals):
        """Return a GroundingBatch containing GroundedObject values."""
