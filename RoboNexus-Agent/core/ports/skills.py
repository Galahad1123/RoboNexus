"""One bounded atomic skill attempt boundary."""

from abc import ABC, abstractmethod


class SkillExecutorPortError(RuntimeError):
    pass


class SkillExecutorUnavailable(SkillExecutorPortError):
    pass


class SkillPreconditionRejected(SkillExecutorPortError):
    def __init__(self, reason, evidence=None):
        self.reason = str(reason)
        self.evidence = dict(evidence or {})
        super().__init__(self.reason)


class SkillExecutorPort(ABC):
    @abstractmethod
    def check_preconditions(self, skill_id):
        """Reject an invalid or already-satisfied backend state."""

    @abstractmethod
    def check_available(self, skill_id):
        """Validate routing, connection, handshake and endpoint health."""

    @abstractmethod
    def execute(
        self, skill_id, groundings, attempt, logical_subtask_id, step_limit
    ):
        """Execute once in the current scene and return a structured result."""
