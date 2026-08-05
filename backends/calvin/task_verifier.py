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
        self.success_guard = None
        self.success_guard_streak = 0

    def begin(self, episode, start_info):
        self.start_info = start_info
        self.oracle_task = episode.oracle_task
        self.success_guard = getattr(episode, "success_guard", None)
        self.success_guard_streak = 0

    def evaluate(self, current_info, observation=None, executed_action=None):
        achieved = self.task_oracle.get_task_info_for_set(
            self.start_info, current_info, {self.oracle_task}
        )
        oracle_success = self.oracle_task in achieved
        guarded_success = oracle_success and self._guard_satisfied(
            observation, executed_action
        )
        self.success_guard_streak = (
            self.success_guard_streak + 1 if guarded_success else 0
        )
        required_steps = int((self.success_guard or {}).get("consecutive_steps", 1))
        success = self.success_guard_streak >= required_steps
        return VerificationResult(
            status="success" if success else "unknown",
            reason=(
                "oracle_predicate_satisfied"
                if success
                else "success_guard_not_satisfied"
                if oracle_success
                else "not_yet_satisfied"
            ),
            confidence=1.0,
            evidence={
                "oracle_task": self.oracle_task,
                "oracle_satisfied": oracle_success,
                "success_guard_streak": self.success_guard_streak,
                "required_success_steps": required_steps,
            },
        )

    def _guard_satisfied(self, observation, executed_action):
        if not self.success_guard:
            return True
        if observation is None or executed_action is None:
            return False
        from calvin.evaluation.remote_policy.rollout_engine import (
            success_guard_satisfied,
        )

        return success_guard_satisfied(
            self.success_guard, observation, executed_action
        )
