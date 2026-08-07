"""Whole-goal CALVIN Oracle verifier."""

from core.domain import GoalVerificationResult
from core.ports import GoalVerifierPort, GoalVerifierUnavailable


class CalvinOracleGoalVerifier(GoalVerifierPort):
    """Verify every required goal predicate against the goal-start scene."""

    def __init__(self, adapter, task_oracle, start_info=None):
        self.adapter = adapter
        self.task_oracle = task_oracle
        try:
            self.start_info = start_info if start_info is not None else adapter.get_info()
        except Exception as exc:
            raise GoalVerifierUnavailable(
                "cannot capture CALVIN goal start state: {}: {}".format(
                    type(exc).__name__, exc
                )
            )

    def verify(self, goal, observation, use_goal_reference=True):
        del observation, use_goal_reference
        try:
            predicates = _predicate_names(goal.success_spec)
            current_info = self.adapter.get_info()
            achieved = set(
                self.task_oracle.get_task_info_for_set(
                    self.start_info, current_info, set(predicates)
                )
            )
        except GoalVerifierUnavailable:
            raise
        except Exception as exc:
            raise GoalVerifierUnavailable(
                "CALVIN Oracle goal verification failed: {}: {}".format(
                    type(exc).__name__, exc
                )
            )
        evidence = {
            "predicates": [
                {"name": name, "satisfied": name in achieved}
                for name in predicates
            ]
        }
        success = all(name in achieved for name in predicates)
        return GoalVerificationResult(
            status="success" if success else "failure",
            backend="calvin_oracle",
            reason=(
                "all_required_predicates_satisfied"
                if success
                else "required_predicates_not_satisfied"
            ),
            confidence=1.0,
            evidence=evidence,
        )


def _predicate_names(success_spec):
    if not isinstance(success_spec, dict):
        raise ValueError("success_spec must be a mapping")
    raw = success_spec.get("predicates")
    if not isinstance(raw, list) or not raw:
        raise ValueError("success_spec.predicates must be a non-empty array")
    result = []
    for index, predicate in enumerate(raw):
        name = predicate.get("name") if isinstance(predicate, dict) else predicate
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "success_spec.predicates[{}] requires a non-empty name".format(
                    index
                )
            )
        result.append(name.strip())
    if len(set(result)) != len(result):
        raise ValueError("success_spec.predicates must be unique")
    return tuple(result)
