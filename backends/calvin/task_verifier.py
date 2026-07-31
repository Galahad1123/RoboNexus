"""Replaceable task verification interface and CALVIN-oracle implementation."""

from dataclasses import dataclass, field


@dataclass
class VerificationResult:
    status: str
    reason: str
    confidence: object = None
    evidence: dict = field(default_factory=dict)

    @property
    def success(self):
        return self.status == "success"


class CalvinOracleTaskVerifier:
    """Use CALVIN predicates without exposing them to the retry controller."""

    def __init__(self, task_oracle):
        self.task_oracle = task_oracle
        self.start_info = None
        self.oracle_task = None

    def begin(self, episode, start_info):
        self.start_info = start_info
        self.oracle_task = episode.oracle_task

    def evaluate(self, current_info):
        achieved = self.task_oracle.get_task_info_for_set(
            self.start_info, current_info, {self.oracle_task}
        )
        success = self.oracle_task in achieved
        return VerificationResult(
            status="success" if success else "unknown",
            reason="oracle_predicate_satisfied" if success else "not_yet_satisfied",
            confidence=1.0,
            evidence={"oracle_task": self.oracle_task},
        )
