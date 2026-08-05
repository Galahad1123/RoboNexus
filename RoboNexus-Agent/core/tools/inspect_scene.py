"""Backend-neutral implementation of the ``inspect_scene`` Agent tool."""

from core.domain import (
    ErrorCode,
    ToolError,
    ToolResultEnvelope,
    ToolStatus,
)
from core.ports import EnvironmentPortError
from core.runtime import SessionRejected


class InspectSceneTool:
    name = "inspect_scene"
    description = (
        "Capture the latest fixed third-person and wrist-camera observations "
        "without changing the robot environment."
    )
    parameters = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why a fresh observation is needed.",
            },
            "include_robot_summary": {"type": "boolean", "default": True},
            "include_facility_summary": {"type": "boolean", "default": True},
        },
        "additionalProperties": False,
    }

    def __init__(self, session, environment, artifacts):
        self.session = session
        self.environment = environment
        self.artifacts = artifacts

    def call(self, params=None, tool_call_id=None):
        tool_call_id = tool_call_id or self.session.next_tool_call_id()
        try:
            replay = self.session.start_tool_call(tool_call_id, self.name)
        except SessionRejected as exc:
            return self._standalone_result(
                tool_call_id, ToolStatus.REJECTED, error=exc.error
            )
        if replay is not None:
            return replay

        try:
            parsed = _validate_params(params or {})
        except (TypeError, ValueError) as exc:
            result = self.session.result(
                tool_call_id,
                ToolStatus.REJECTED,
                error=ToolError(
                    ErrorCode.INVALID_ARGUMENT,
                    str(exc),
                    retryable=True,
                ),
            )
            self.session.finish_tool_call(result)
            return result

        try:
            capture = self.environment.capture_scene()
            capture.validate()
            self._synchronize_environment_steps(capture.environment_steps)
            static_artifact = self.artifacts.save_image(
                self.session.session_id, "current_static", capture.static_rgb
            )
            wrist_artifact = self.artifacts.save_image(
                self.session.session_id, "current_wrist", capture.wrist_rgb
            )
            static_artifact.validate()
            wrist_artifact.validate()
            observation = self.session.add_observation(
                static_artifact.uri, wrist_artifact.uri
            )
            data = {
                "reason": parsed["reason"],
                "observation": _observation_dict(observation),
            }
            if parsed["include_robot_summary"]:
                data["robot_summary"] = dict(capture.robot_summary)
            if parsed["include_facility_summary"]:
                data["facility_summary"] = dict(capture.facility_summary)
            result = self.session.result(
                tool_call_id,
                ToolStatus.SUCCEEDED,
                data=data,
                artifacts=(static_artifact, wrist_artifact),
            )
        except SessionRejected as exc:
            result = self.session.result(
                tool_call_id,
                ToolStatus.REJECTED,
                error=exc.error,
            )
        except (EnvironmentPortError, OSError) as exc:
            result = self.session.result(
                tool_call_id,
                ToolStatus.UNAVAILABLE,
                error=ToolError(
                    ErrorCode.ENVIRONMENT_UNAVAILABLE,
                    "{}: {}".format(type(exc).__name__, exc),
                    retryable=True,
                ),
            )
        except Exception as exc:
            result = self.session.result(
                tool_call_id,
                ToolStatus.UNAVAILABLE,
                error=ToolError(
                    ErrorCode.TOOL_INTERNAL_ERROR,
                    "{}: {}".format(type(exc).__name__, exc),
                    retryable=True,
                ),
            )
        self.session.finish_tool_call(result)
        return result

    def _synchronize_environment_steps(self, backend_steps):
        session_steps = self.session.environment_steps_used
        if backend_steps < session_steps:
            raise EnvironmentPortError(
                "environment step counter moved backwards: {} < {}".format(
                    backend_steps, session_steps
                )
            )
        delta = backend_steps - session_steps
        if delta:
            self.session.add_environment_steps(delta)

    def _standalone_result(self, call_id, status, error=None):
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
        raise TypeError("inspect_scene parameters must be a mapping")
    allowed = {"reason", "include_robot_summary", "include_facility_summary"}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError("unknown inspect_scene parameters: {}".format(unknown))
    reason = params.get("reason", "refresh current scene")
    if not isinstance(reason, str):
        raise TypeError("reason must be a string")
    include_robot = params.get("include_robot_summary", True)
    include_facility = params.get("include_facility_summary", True)
    if not isinstance(include_robot, bool):
        raise TypeError("include_robot_summary must be a boolean")
    if not isinstance(include_facility, bool):
        raise TypeError("include_facility_summary must be a boolean")
    return {
        "reason": reason,
        "include_robot_summary": include_robot,
        "include_facility_summary": include_facility,
    }


def _observation_dict(observation):
    return {
        "observation_id": observation.observation_id,
        "environment_revision": observation.environment_revision,
        "captured_at_step": observation.captured_at_step,
        "static_image": observation.static_image,
        "wrist_image": observation.wrist_image,
    }
