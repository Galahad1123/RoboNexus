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

    def notify_recovery(self, failed_skill_id, result):
        """Optionally preserve backend recovery state for the next retry."""

        del failed_skill_id, result
