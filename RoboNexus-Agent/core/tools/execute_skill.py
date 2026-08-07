"""Backend-neutral implementation of the ``execute_skill`` Agent tool."""

from core.domain import ErrorCode, ToolError, ToolResultEnvelope, ToolStatus
from core.ports import (
    EnvironmentPortError,
    SkillExecutorPortError,
    SkillExecutorUnavailable,
    SkillPreconditionRejected,
)
from core.runtime import SessionRejected


class ExecuteSkillTool:
    name = "execute_skill"
    description = (
        "Execute exactly one bounded atomic robot-skill attempt in the current "
        "scene using Registry-controlled policy routing and limits."
    )
    parameters = {
        "type": "object",
        "required": ["skill_id", "grounding_ids", "intent"],
        "properties": {
            "skill_id": {"type": "string"},
            "grounding_ids": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
            "intent": {
                "type": "string",
                "description": "Why this atomic skill advances the user goal.",
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, session, skill_registry, executor, artifacts):
        self.session = session
        self.skill_registry = skill_registry
        self.executor = executor
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
            skill = self._get_skill(parsed["skill_id"])
            groundings = tuple(
                self.session.get_grounding(grounding_id, require_current=True)
                for grounding_id in parsed["grounding_ids"]
            )
            _validate_role_contract(skill, groundings)
            _validate_distinct_instances(groundings)
            before = self.session.latest_observation
            logical_subtask_id = self.session.resolve_logical_subtask(
                skill.skill_id, groundings
            )
            self.executor.check_preconditions(skill.skill_id)
            self.executor.check_available(skill.skill_id)
            if self.session.attempts_used(logical_subtask_id) >= skill.max_attempts:
                raise SessionRejected(
                    ErrorCode.ATTEMPT_BUDGET_EXHAUSTED,
                    "skill attempt budget exhausted for {!r}".format(
                        logical_subtask_id
                    ),
                    retryable=False,
                )
            environment_steps_remaining = (
                self.session.budget_snapshot().environment_steps_remaining
            )
            if environment_steps_remaining < 1:
                raise SessionRejected(
                    ErrorCode.GOAL_BUDGET_EXHAUSTED,
                    "environment step budget exhausted",
                    retryable=False,
                )
            step_limit = min(skill.max_steps, environment_steps_remaining)
            attempt = self.session.consume_attempt(logical_subtask_id)
            execution = self.executor.execute(
                skill.skill_id,
                groundings,
                attempt,
                logical_subtask_id,
                step_limit,
            )
            execution.validate()
            self._synchronize_environment_steps(
                execution.scene_capture.environment_steps
            )
            static_artifact = self.artifacts.save_image(
                self.session.session_id,
                "after_{}_static".format(logical_subtask_id),
                execution.scene_capture.static_rgb,
            )
            wrist_artifact = self.artifacts.save_image(
                self.session.session_id,
                "after_{}_wrist".format(logical_subtask_id),
                execution.scene_capture.wrist_rgb,
            )
            artifacts = [static_artifact, wrist_artifact]
            if execution.frames:
                artifacts.append(
                    self.artifacts.save_video(
                        self.session.session_id,
                        "{}_attempt_{:02d}".format(logical_subtask_id, attempt),
                        execution.frames,
                        execution.video_fps,
                    )
                )
            for artifact in artifacts:
                artifact.validate()
            after = self.session.add_observation(
                static_artifact.uri, wrist_artifact.uri
            )
            data = {
                "skill_id": skill.skill_id,
                "logical_subtask_id": logical_subtask_id,
                "attempt": attempt,
                "policy_id": execution.policy_id,
                "success": execution.success,
                "termination_reason": execution.termination_reason,
                "steps": execution.steps,
                "verification": dict(execution.verification),
                "scene_continuation": True,
                "before_observation_id": before.observation_id,
                "after_observation_id": after.observation_id,
                "after_observation": _observation_dict(after),
                "state_delta": dict(execution.state_delta),
                "recommended_actions": list(execution.recommended_actions),
                "intent": parsed["intent"],
            }
            result = self.session.result(
                tool_call_id,
                ToolStatus.SUCCEEDED if execution.success else ToolStatus.FAILED,
                data=data,
                artifacts=tuple(artifacts),
            )
        except (TypeError, ValueError) as exc:
            result = self._error_result(
                tool_call_id, ToolStatus.REJECTED, ErrorCode.INVALID_ARGUMENT, exc, True
            )
        except SessionRejected as exc:
            result = self.session.result(
                tool_call_id, ToolStatus.REJECTED, error=exc.error
            )
        except SkillExecutorUnavailable as exc:
            result = self._error_result(
                tool_call_id, ToolStatus.UNAVAILABLE, ErrorCode.POLICY_UNAVAILABLE, exc, True
            )
        except SkillPreconditionRejected as exc:
            result = self.session.result(
                tool_call_id,
                ToolStatus.REJECTED,
                error=ToolError(
                    ErrorCode.INVALID_PRECONDITION,
                    exc.reason,
                    retryable=True,
                    details=exc.evidence,
                ),
            )
        except (SkillExecutorPortError, EnvironmentPortError, OSError) as exc:
            result = self._error_result(
                tool_call_id, ToolStatus.UNAVAILABLE, ErrorCode.TOOL_INTERNAL_ERROR, exc, True
            )
        except Exception as exc:
            result = self._error_result(
                tool_call_id, ToolStatus.UNAVAILABLE, ErrorCode.TOOL_INTERNAL_ERROR, exc, True
            )
        self.session.finish_tool_call(result)
        return result

    def _get_skill(self, skill_id):
        try:
            return self.skill_registry.get(skill_id)
        except KeyError:
            raise SessionRejected(
                ErrorCode.UNKNOWN_SKILL,
                "unknown skill_id {!r}".format(skill_id),
                retryable=True,
            )

    def _synchronize_environment_steps(self, backend_steps):
        current = self.session.environment_steps_used
        if backend_steps < current:
            raise EnvironmentPortError(
                "environment step counter moved backwards: {} < {}".format(
                    backend_steps, current
                )
            )
        self.session.add_environment_steps(backend_steps - current)

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
        raise TypeError("execute_skill parameters must be a mapping")
    required = {"skill_id", "grounding_ids", "intent"}
    missing = sorted(required - set(params))
    unknown = sorted(set(params) - required)
    if missing:
        raise ValueError("missing execute_skill parameters: {}".format(missing))
    if unknown:
        raise ValueError("unknown execute_skill parameters: {}".format(unknown))
    skill_id = _nonempty(params["skill_id"], "skill_id")
    intent = _nonempty(params["intent"], "intent")
    grounding_ids = params["grounding_ids"]
    if not isinstance(grounding_ids, list):
        raise TypeError("grounding_ids must be an array")
    if len(set(grounding_ids)) != len(grounding_ids):
        raise ValueError("grounding_ids must be unique")
    grounding_ids = tuple(
        _nonempty(item, "grounding_ids[]") for item in grounding_ids
    )
    return {
        "skill_id": skill_id,
        "grounding_ids": grounding_ids,
        "intent": intent,
    }


def _validate_role_contract(skill, groundings):
    actual = tuple(sorted(grounding.role.value for grounding in groundings))
    expected = tuple(sorted(skill.grounding_roles))
    if actual != expected:
        raise SessionRejected(
            ErrorCode.INVALID_PRECONDITION,
            "skill {!r} requires grounding roles {}, got {}".format(
                skill.skill_id, list(expected), list(actual)
            ),
            retryable=True,
        )


def _validate_distinct_instances(groundings):
    by_role = {grounding.role.value: grounding for grounding in groundings}
    if "source" not in by_role or "destination" not in by_role:
        return
    source = by_role["source"]
    destination = by_role["destination"]
    source_instance = source.metadata.get("instance_id") or source.semantic_id
    destination_instance = (
        destination.metadata.get("instance_id") or destination.semantic_id
    )
    if source_instance is not None and source_instance == destination_instance:
        raise SessionRejected(
            ErrorCode.INVALID_PRECONDITION,
            "source and destination resolve to the same scene instance",
            retryable=True,
        )


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
