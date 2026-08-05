"""Target-approach recovery boundary."""

from abc import ABC, abstractmethod


class ApproachControllerPort(ABC):
    @abstractmethod
    def approach(self, grounding, xy_threshold_m):
        """Move near a grounded target and return a structured motion result."""

