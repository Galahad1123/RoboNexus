"""Backend-neutral implementation of the ``approach_target`` Agent tool."""

import math

from core.domain import ErrorCode, ToolError, ToolResultEnvelope, ToolStatus
from core.ports import (
    ApproachControllerPortError,
    ApproachRejected,
    ApproachUnavailable,
    EnvironmentPortError,
)
from core.runtime import SessionRejected


class ApproachTargetTool:
    name = "approach_target"
    description = (
        "Move the end effector near one current grounded target in XY without "
        "performing the manipulation skill itself."
    )
    parameters = {
        "type": "object",
        "required": ["grounding_id", "intent"],
        "properties": {
            "grounding_id": {"type": "string"},
            "intent": {"type": "string"},
            "xy_threshold_m": {
                "type": "number",
                "minimum": 0.02,
                "maximum": 0.20,
                "default": 0.08,
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, session, controller, artifacts):
        self.session = session
        self.controller = controller
        self.artifacts = artifacts

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
            grounding = self.session.get_grounding(
                parsed["grounding_id"], require_current=True
            )
            before = self.session.latest_observation
            remaining = self.session.budget_snapshot().environment_steps_remaining
            if remaining < 1:
                raise SessionRejected(
                    ErrorCode.GOAL_BUDGET_EXHAUSTED,
                    "environment step budget exhausted",
                    retryable=False,
                )
            approach = self.controller.approach(
                grounding, parsed["xy_threshold_m"], remaining
            )
            approach.validate()
            self._synchronize_environment_steps(
                approach.scene_capture.environment_steps, approach.steps
            )
            static_artifact = self.artifacts.save_image(
                self.session.session_id,
                "approach_{}_static".format(grounding.grounding_id),
                approach.scene_capture.static_rgb,
            )
            wrist_artifact = self.artifacts.save_image(
                self.session.session_id,
                "approach_{}_wrist".format(grounding.grounding_id),
                approach.scene_capture.wrist_rgb,
            )
            static_artifact.validate()
            wrist_artifact.validate()
            after = self.session.add_observation(
                static_artifact.uri, wrist_artifact.uri
            )
            data = {
                "grounding_id": grounding.grounding_id,
                "success": approach.success,
                "reason": approach.reason,
                "steps": approach.steps,
                "initial_xy_distance_m": approach.initial_xy_distance_m,
                "final_xy_distance_m": approach.final_xy_distance_m,
                "threshold_m": approach.threshold_m,
                "before_observation_id": before.observation_id,
                "after_observation_id": after.observation_id,
                "after_observation": _observation_dict(after),
                "scene_continuation": True,
                "intent": parsed["intent"],
            }
            result = self.session.result(
                tool_call_id,
                ToolStatus.SUCCEEDED if approach.success else ToolStatus.FAILED,
                data=data,
                artifacts=(static_artifact, wrist_artifact),
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
        except ApproachRejected as exc:
            result = self._error_result(
                tool_call_id,
                ToolStatus.REJECTED,
                ErrorCode.UNSUPPORTED_GROUNDING,
                exc,
                True,
            )
        except ApproachUnavailable as exc:
            result = self._error_result(
                tool_call_id,
                ToolStatus.UNAVAILABLE,
                ErrorCode.ENVIRONMENT_UNAVAILABLE,
                exc,
                True,
            )
        except (ApproachControllerPortError, EnvironmentPortError, OSError) as exc:
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

    def _synchronize_environment_steps(self, backend_steps, reported_steps):
        current = self.session.environment_steps_used
        if backend_steps < current:
            raise EnvironmentPortError(
                "environment step counter moved backwards: {} < {}".format(
                    backend_steps, current
                )
            )
        delta = backend_steps - current
        if delta != reported_steps:
            raise EnvironmentPortError(
                "approach reported {} steps but backend advanced {}".format(
                    reported_steps, delta
                )
            )
        self.session.add_environment_steps(delta)

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
        raise TypeError("approach_target parameters must be a mapping")
    required = {"grounding_id", "intent"}
    allowed = required | {"xy_threshold_m"}
    missing = sorted(required - set(params))
    unknown = sorted(set(params) - allowed)
    if missing:
        raise ValueError("missing approach_target parameters: {}".format(missing))
    if unknown:
        raise ValueError("unknown approach_target parameters: {}".format(unknown))
    grounding_id = _nonempty(params["grounding_id"], "grounding_id")
    intent = _nonempty(params["intent"], "intent")
    threshold = params.get("xy_threshold_m", 0.08)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0.02 <= threshold <= 0.20
    ):
        raise ValueError("xy_threshold_m must be a finite number in [0.02, 0.20]")
    return {
        "grounding_id": grounding_id,
        "intent": intent,
        "xy_threshold_m": float(threshold),
    }


def _nonempty(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(path))
    return value.strip()


def _observation_dict(observation):
    return {
        "observation_id": observation.observation_id,
        "environment_revision": observation.environment_revision,
        "captured_at_step": observation.captured_at_step,
        "static_image": observation.static_image,
        "wrist_image": observation.wrist_image,
    }
