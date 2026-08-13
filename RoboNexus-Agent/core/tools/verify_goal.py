"""Backend-neutral implementation of the ``verify_goal`` Agent tool."""

from core.domain import (
    ErrorCode,
    GoalVerificationResult,
    ToolError,
    ToolResultEnvelope,
    ToolStatus,
)
from core.ports import GoalVerifierPortError, GoalVerifierUnavailable
from core.runtime import SessionRejected


class VerifyGoalTool:
    name = "verify_goal"
    description = (
        "Verify the complete user goal against its immutable success specification."
    )
    parameters = {
        "type": "object",
        "required": ["claim", "observation_id"],
        "properties": {
            "claim": {
                "type": "string",
                "description": "Final state believed to be satisfied.",
            },
            "observation_id": {"type": "string"},
            "use_goal_reference": {"type": "boolean", "default": True},
        },
        "additionalProperties": False,
    }

    def __init__(self, session, verifier):
        self.session = session
        self.verifier = verifier

    def call(self, params=None, tool_call_id=None):
        tool_call_id = tool_call_id or self.session.next_tool_call_id()
        try:
            replay = self.session.start_tool_call(tool_call_id, self.name)
        except SessionRejected as exc:
            return self._standalone_result(tool_call_id, ToolStatus.REJECTED, exc.error)
        if replay is not None:
            return replay

        try:
            parsed = _validate_params(params)
            observation = self.session.get_observation(
                parsed["observation_id"], require_current=True
            )
            verification = self.verifier.verify(
                self.session.goal,
                observation,
                use_goal_reference=parsed["use_goal_reference"],
            )
            verification = _apply_sequence_achievements(
                self.session, verification
            )
            verification.validate()
            reference_used = bool(
                parsed["use_goal_reference"]
                and self.session.goal.target_reference_image
            )
            data = {
                "goal_id": self.session.goal.goal_id,
                "status": verification.status,
                "backend": verification.backend,
                "reason": verification.reason,
                "confidence": verification.confidence,
                "observation_id": observation.observation_id,
                "goal_reference_used": reference_used,
                "goal_reference_image": (
                    self.session.goal.target_reference_image
                    if reference_used
                    else None
                ),
                "evidence": dict(verification.evidence),
                "claim": parsed["claim"],
            }
            self.session.record_verification(
                verification.status,
                verification.reason,
                verification.evidence,
            )
            result = self.session.result(
                tool_call_id, ToolStatus.SUCCEEDED, data=data
            )
            if verification.status == "success":
                self.session.mark_succeeded(
                    verification.status,
                    verification.reason,
                    details={
                        "backend": verification.backend,
                        "observation_id": observation.observation_id,
                        "evidence": dict(verification.evidence),
                    },
                )
        except (TypeError, ValueError) as exc:
            result = self._error_result(
                tool_call_id,
                ToolStatus.REJECTED,
                ErrorCode.INVALID_ARGUMENT,
                exc,
                True,
            )
        except SessionRejected as exc:
            result = self.session.result(
                tool_call_id, ToolStatus.REJECTED, error=exc.error
            )
        except GoalVerifierUnavailable as exc:
            result = self._error_result(
                tool_call_id,
                ToolStatus.UNAVAILABLE,
                ErrorCode.VERIFIER_UNAVAILABLE,
                exc,
                True,
            )
        except GoalVerifierPortError as exc:
            result = self._error_result(
                tool_call_id,
                ToolStatus.UNAVAILABLE,
                ErrorCode.TOOL_INTERNAL_ERROR,
                exc,
                True,
            )
        except Exception as exc:
            result = self._error_result(
                tool_call_id,
                ToolStatus.UNAVAILABLE,
                ErrorCode.TOOL_INTERNAL_ERROR,
                exc,
                True,
            )
        self.session.finish_tool_call(result)
        return result

    def _error_result(self, call_id, status, code, exc, retryable):
        return self.session.result(
            call_id,
            status,
            error=ToolError(
                code,
                "{}: {}".format(type(exc).__name__, exc),
                retryable=retryable,
            ),
        )

    def _standalone_result(self, call_id, status, error):
        return ToolResultEnvelope(
            tool_call_id=call_id,
            session_id=self.session.session_id,
            status=status,
            data={},
            budget=self.session.budget_snapshot(),
            error=error,
        )


def _validate_params(params):
    if not isinstance(params, dict):
        raise TypeError("verify_goal parameters must be a mapping")
    required = {"claim", "observation_id"}
    allowed = required | {"use_goal_reference"}
    missing = sorted(required - set(params))
    unknown = sorted(set(params) - allowed)
    if missing:
        raise ValueError("missing verify_goal parameters: {}".format(missing))
    if unknown:
        raise ValueError("unknown verify_goal parameters: {}".format(unknown))
    use_reference = params.get("use_goal_reference", True)
    if not isinstance(use_reference, bool):
        raise TypeError("use_goal_reference must be boolean")
    return {
        "claim": _nonempty(params["claim"], "claim"),
        "observation_id": _nonempty(params["observation_id"], "observation_id"),
        "use_goal_reference": use_reference,
    }


def _nonempty(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(path))
    return value.strip()


def _apply_sequence_achievements(session, verification):
    """Latch Oracle-confirmed atomic skills for ordered CALVIN goals.

    Transient intermediate predicates such as ``lift_*`` need not remain true
    after a later placement. This mode is explicit so ordinary final-state
    goals continue to use only the live verifier result.
    """
    if session.goal.success_spec.get("mode") != "skill_sequence":
        return verification
    predicates = session.goal.success_spec.get("predicates", ())
    names = [
        item.get("name") if isinstance(item, dict) else item
        for item in predicates
    ]
    achieved = set(session.achieved_skill_ids)
    achievement_sequence = list(session.achievement_sequence)
    live = {
        item.get("name"): bool(item.get("satisfied"))
        for item in verification.evidence.get("predicates", ())
        if isinstance(item, dict)
    }
    evidence = {
        "predicates": [
            {
                "name": name,
                "satisfied": bool(live.get(name, False) or name in achieved),
                "latched_skill_achievement": name in achieved,
            }
            for name in names
        ],
        "achieved_skill_ids": sorted(achieved),
        "achievement_sequence": achievement_sequence,
        "live_backend_evidence": dict(verification.evidence),
    }
    sequence_complete = achievement_sequence[: len(names)] == names
    evidence["required_sequence"] = names
    evidence["sequence_order_satisfied"] = sequence_complete
    success = bool(names) and sequence_complete
    return GoalVerificationResult(
        status="success" if success else "failure",
        backend="{}+skill_sequence".format(verification.backend),
        reason=(
            "all_sequence_skills_achieved"
            if success
            else "sequence_skills_not_yet_achieved"
        ),
        confidence=verification.confidence,
        evidence=evidence,
    )
