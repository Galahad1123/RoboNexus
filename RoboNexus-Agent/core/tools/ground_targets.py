"""Backend-neutral implementation of the ``ground_targets`` Agent tool."""

import math

from core.domain import ErrorCode, Role, ToolError, ToolResultEnvelope, ToolStatus
from core.ports import (
    GrounderPortError,
    GroundingNotFoundError,
    UnsupportedGroundingError,
)
from core.runtime import SessionRejected


class GroundTargetsTool:
    name = "ground_targets"
    description = (
        "Bind the source, destination, or target selected by the planner to "
        "objects in a current robot observation."
    )
    parameters = {
        "type": "object",
        "required": ["skill_id", "observation_id", "proposals"],
        "properties": {
            "skill_id": {"type": "string"},
            "observation_id": {"type": "string"},
            "proposals": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "required": ["role", "representation", "description"],
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": ["source", "destination", "target"],
                        },
                        "representation": {
                            "type": "string",
                            "enum": ["semantic", "bbox", "point"],
                        },
                        "semantic_id": {"type": "string"},
                        "camera": {
                            "type": "string",
                            "enum": ["static", "wrist"],
                        },
                        "bbox_xyxy": {
                            "type": "array",
                            "minItems": 4,
                            "maxItems": 4,
                            "items": {"type": "number"},
                        },
                        "point_xy": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "number"},
                        },
                        "description": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, session, skill_registry, grounder):
        self.session = session
        self.skill_registry = skill_registry
        self.grounder = grounder

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
            if parsed["skill_id"] not in self.skill_registry:
                raise SessionRejected(
                    ErrorCode.UNKNOWN_SKILL,
                    "unknown skill_id {!r}".format(parsed["skill_id"]),
                    retryable=True,
                )
            observation = self.session.get_observation(
                parsed["observation_id"], require_current=True
            )
            proposals = []
            for proposal in parsed["proposals"]:
                item = dict(proposal)
                item["grounding_id"] = self.session.next_grounding_id()
                proposals.append(item)
            batch = self.grounder.ground(parsed["skill_id"], observation, proposals)
            self._validate_batch(batch, observation, proposals)
            for grounding in batch.groundings:
                self.session.add_grounding(grounding)
            data = {
                "skill_id": parsed["skill_id"],
                "observation_id": observation.observation_id,
                "groundings": [_grounding_dict(item) for item in batch.groundings],
                "preview": batch.preview.uri,
            }
            result = self.session.result(
                tool_call_id,
                ToolStatus.SUCCEEDED,
                data=data,
                artifacts=tuple(batch.artifacts) + (batch.preview,),
            )
        except (TypeError, ValueError) as exc:
            result = self._error_result(
                tool_call_id, ErrorCode.INVALID_ARGUMENT, exc, retryable=True
            )
        except SessionRejected as exc:
            result = self.session.result(
                tool_call_id, ToolStatus.REJECTED, error=exc.error
            )
        except UnsupportedGroundingError as exc:
            result = self._error_result(
                tool_call_id, ErrorCode.UNSUPPORTED_GROUNDING, exc, retryable=True
            )
        except GroundingNotFoundError as exc:
            result = self._error_result(
                tool_call_id, ErrorCode.GROUNDING_NOT_FOUND, exc, retryable=True
            )
        except GrounderPortError as exc:
            result = self._error_result(
                tool_call_id, ErrorCode.TOOL_INTERNAL_ERROR, exc, retryable=True,
                status=ToolStatus.UNAVAILABLE,
            )
        except Exception as exc:
            result = self._error_result(
                tool_call_id, ErrorCode.TOOL_INTERNAL_ERROR, exc, retryable=True,
                status=ToolStatus.UNAVAILABLE,
            )
        self.session.finish_tool_call(result)
        return result

    def _validate_batch(self, batch, observation, proposals):
        if batch is None or not hasattr(batch, "groundings"):
            raise GrounderPortError("grounder returned no GroundingBatch")
        if len(batch.groundings) != len(proposals):
            raise GrounderPortError("grounder returned the wrong number of groundings")
        expected = {item["grounding_id"]: item for item in proposals}
        for grounding in batch.groundings:
            grounding.validate()
            proposal = expected.get(grounding.grounding_id)
            if proposal is None:
                raise GrounderPortError("grounder returned an unexpected grounding_id")
            if grounding.observation_id != observation.observation_id:
                raise GrounderPortError("grounding observation_id does not match request")
            if grounding.environment_revision != observation.environment_revision:
                raise GrounderPortError("grounding environment revision does not match request")
            if grounding.role.value != proposal["role"]:
                raise GrounderPortError("grounding role does not match proposal")
            if not grounding.mask_handle:
                raise GroundingNotFoundError(
                    "grounding {!r} has no mask".format(grounding.grounding_id)
                )
        if batch.preview is None:
            raise GrounderPortError("grounder returned no preview artifact")
        batch.preview.validate()
        for artifact in batch.artifacts:
            artifact.validate()

    def _error_result(self, call_id, code, exc, retryable, status=ToolStatus.REJECTED):
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
        raise TypeError("ground_targets parameters must be a mapping")
    required = {"skill_id", "observation_id", "proposals"}
    missing = sorted(required - set(params))
    unknown = sorted(set(params) - required)
    if missing:
        raise ValueError("missing ground_targets parameters: {}".format(missing))
    if unknown:
        raise ValueError("unknown ground_targets parameters: {}".format(unknown))
    skill_id = _nonempty(params["skill_id"], "skill_id")
    observation_id = _nonempty(params["observation_id"], "observation_id")
    raw_proposals = params["proposals"]
    if not isinstance(raw_proposals, list) or not 1 <= len(raw_proposals) <= 2:
        raise ValueError("proposals must contain one or two entries")
    proposals = [_validate_proposal(item, index) for index, item in enumerate(raw_proposals)]
    roles = [item["role"] for item in proposals]
    if len(set(roles)) != len(roles):
        raise ValueError("proposal roles must be unique")
    return {"skill_id": skill_id, "observation_id": observation_id, "proposals": proposals}


def _validate_proposal(proposal, index):
    path = "proposals[{}]".format(index)
    if not isinstance(proposal, dict):
        raise TypeError("{} must be a mapping".format(path))
    allowed = {
        "role", "representation", "semantic_id", "camera",
        "bbox_xyxy", "point_xy", "description",
    }
    unknown = sorted(set(proposal) - allowed)
    if unknown:
        raise ValueError("{} has unknown fields: {}".format(path, unknown))
    role = _nonempty(proposal.get("role"), "{}.role".format(path))
    if role not in {item.value for item in Role}:
        raise ValueError("{}.role is unsupported".format(path))
    representation = _nonempty(
        proposal.get("representation"), "{}.representation".format(path)
    )
    if representation not in {"semantic", "bbox", "point"}:
        raise ValueError("{}.representation is unsupported".format(path))
    result = {
        "role": role,
        "representation": representation,
        "description": _nonempty(
            proposal.get("description"), "{}.description".format(path)
        ),
    }
    if representation == "semantic":
        result["semantic_id"] = _nonempty(
            proposal.get("semantic_id"), "{}.semantic_id".format(path)
        )
    else:
        camera = proposal.get("camera")
        if camera not in {"static", "wrist"}:
            raise ValueError("{}.camera must be static or wrist".format(path))
        result["camera"] = camera
        field = "bbox_xyxy" if representation == "bbox" else "point_xy"
        expected = 4 if representation == "bbox" else 2
        values = proposal.get(field)
        if not isinstance(values, list) or len(values) != expected:
            raise ValueError("{}.{} must contain {} numbers".format(path, field, expected))
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise TypeError("{}.{} must contain finite numbers".format(path, field))
        values = tuple(float(value) for value in values)
        if field == "bbox_xyxy" and not (values[0] < values[2] and values[1] < values[3]):
            raise ValueError("{}.bbox_xyxy must satisfy x1 < x2 and y1 < y2".format(path))
        result[field] = values
    return result


def _nonempty(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(path))
    return value.strip()


def _grounding_dict(grounding):
    return {
        "grounding_id": grounding.grounding_id,
        "role": grounding.role.value,
        "semantic_id": grounding.semantic_id,
        "source": grounding.source,
        "mask_handle": grounding.mask_handle,
        "confidence": grounding.confidence,
    }
