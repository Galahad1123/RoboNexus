"""Explicit, backend-neutral Agent goal termination tool."""

from core.domain import (
    AgentSessionStatus,
    ErrorCode,
    ToolError,
    ToolResultEnvelope,
    ToolStatus,
)
from core.runtime import SessionRejected


ABORT_REASON_CODES = frozenset(
    {
        "task_infeasible",
        "required_skill_unavailable",
        "repeated_execution_failure",
        "unsafe_to_continue",
        "insufficient_observation",
    }
)


class AbortGoalTool:
    name = "abort_goal"
    description = (
        "Explicitly terminate an active goal without claiming success or resetting "
        "the environment."
    )
    parameters = {
        "type": "object",
        "required": ["reason_code", "summary"],
        "properties": {
            "reason_code": {
                "type": "string",
                "enum": sorted(ABORT_REASON_CODES),
            },
            "summary": {"type": "string"},
            "last_observation_id": {"type": "string"},
        },
        "additionalProperties": False,
    }

    def __init__(self, session):
        self.session = session

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
            observation_id = (
                parsed["last_observation_id"]
                or self.session.latest_observation.observation_id
            )
            observation = self.session.get_observation(
                observation_id, require_current=True
            )
            if observation.observation_id != self.session.latest_observation.observation_id:
                raise SessionRejected(
                    ErrorCode.STALE_OBSERVATION,
                    "last_observation_id must identify the latest observation",
                    retryable=True,
                )
            data = {
                "goal_id": self.session.goal.goal_id,
                "status": AgentSessionStatus.ABORTED_BY_AGENT.value,
                "reason_code": parsed["reason_code"],
                "summary": parsed["summary"],
                "last_observation_id": observation.observation_id,
                "scene_continuation": True,
            }
            result = self.session.result(
                tool_call_id, ToolStatus.SUCCEEDED, data=data
            )
            self.session.abort(
                AgentSessionStatus.ABORTED_BY_AGENT,
                parsed["reason_code"],
                details={
                    "summary": parsed["summary"],
                    "last_observation_id": observation.observation_id,
                },
            )
        except (TypeError, ValueError) as exc:
            result = self.session.result(
                tool_call_id,
                ToolStatus.REJECTED,
                error=ToolError(
                    ErrorCode.INVALID_ARGUMENT,
                    "{}: {}".format(type(exc).__name__, exc),
                    retryable=True,
                ),
            )
        except SessionRejected as exc:
            result = self.session.result(
                tool_call_id, ToolStatus.REJECTED, error=exc.error
            )
        self.session.finish_tool_call(result)
        return result

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
        raise TypeError("abort_goal parameters must be a mapping")
    required = {"reason_code", "summary"}
    allowed = required | {"last_observation_id"}
    missing = sorted(required - set(params))
    unknown = sorted(set(params) - allowed)
    if missing:
        raise ValueError("missing abort_goal parameters: {}".format(missing))
    if unknown:
        raise ValueError("unknown abort_goal parameters: {}".format(unknown))
    reason_code = _nonempty(params["reason_code"], "reason_code")
    if reason_code not in ABORT_REASON_CODES:
        raise ValueError("unsupported abort reason_code {!r}".format(reason_code))
    last_observation_id = params.get("last_observation_id")
    if last_observation_id is not None:
        last_observation_id = _nonempty(
            last_observation_id, "last_observation_id"
        )
    return {
        "reason_code": reason_code,
        "summary": _nonempty(params["summary"], "summary"),
        "last_observation_id": last_observation_id,
    }


def _nonempty(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(path))
    return value.strip()
