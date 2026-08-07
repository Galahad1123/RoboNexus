"""Target-approach recovery boundary."""

from abc import ABC, abstractmethod


class ApproachControllerPortError(RuntimeError):
    pass


class ApproachRejected(ApproachControllerPortError):
    pass


class ApproachUnavailable(ApproachControllerPortError):
    pass


class ApproachControllerPort(ABC):
    @abstractmethod
    def approach(self, grounding, xy_threshold_m, step_limit):
        """Move near a grounded target and return a structured motion result."""
