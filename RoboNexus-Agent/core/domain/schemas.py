"""Canonical public schemas shared by Qwen tools and robot backends.

This module intentionally stays free of Qwen, CALVIN, model and numpy imports.
It is compatible with the Python 3.8 CALVIN environment.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


PROTOCOL_VERSION = 1


class ToolStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"


class ErrorCode(str, Enum):
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    UNKNOWN_SKILL = "UNKNOWN_SKILL"
    UNSUPPORTED_GROUNDING = "UNSUPPORTED_GROUNDING"
    GROUNDING_NOT_FOUND = "GROUNDING_NOT_FOUND"
    STALE_OBSERVATION = "STALE_OBSERVATION"
    STALE_GROUNDING = "STALE_GROUNDING"
    INVALID_PRECONDITION = "INVALID_PRECONDITION"
    ATTEMPT_BUDGET_EXHAUSTED = "ATTEMPT_BUDGET_EXHAUSTED"
    REPLAN_BUDGET_EXHAUSTED = "REPLAN_BUDGET_EXHAUSTED"
    GOAL_BUDGET_EXHAUSTED = "GOAL_BUDGET_EXHAUSTED"
    POLICY_UNAVAILABLE = "POLICY_UNAVAILABLE"
    ENVIRONMENT_UNAVAILABLE = "ENVIRONMENT_UNAVAILABLE"
    VERIFIER_UNAVAILABLE = "VERIFIER_UNAVAILABLE"
    TOOL_INTERNAL_ERROR = "TOOL_INTERNAL_ERROR"
    SESSION_TERMINAL = "SESSION_TERMINAL"
    TOOL_CALL_IN_PROGRESS = "TOOL_CALL_IN_PROGRESS"


class AgentSessionStatus(str, Enum):
    ACTIVE = "active"
    WAITING_FOR_USER = "waiting_for_user"
    SUCCEEDED = "succeeded"
    ABORTED_BY_AGENT = "aborted_by_agent"
    ABORTED_BY_USER = "aborted_by_user"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DEPENDENCY_FAILURE = "dependency_failure"
    SAFETY_STOP = "safety_stop"

    @property
    def terminal(self):
        return self not in {self.ACTIVE, self.WAITING_FOR_USER}


class Role(str, Enum):
    SOURCE = "source"
    DESTINATION = "destination"
    TARGET = "target"


class GroundingRepresentation(str, Enum):
    SEMANTIC = "semantic"
    BBOX = "bbox"
    POINT = "point"
    MASK = "mask"


@dataclass(frozen=True)
class RuntimeLimits:
    skill_attempt_limit: int = 3
    replan_limit: int = 2
    tool_call_limit: int = 32
    environment_step_limit: int = 2000

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("{} must be a positive integer".format(name))


@dataclass(frozen=True)
class ArtifactRef:
    kind: str
    uri: str
    mime_type: str

    def validate(self):
        _nonempty(self.kind, "artifact.kind")
        _nonempty(self.uri, "artifact.uri")
        _nonempty(self.mime_type, "artifact.mime_type")


@dataclass(frozen=True)
class ObservationRef:
    observation_id: str
    environment_revision: int
    captured_at_step: int
    static_image: str
    wrist_image: str

    def validate(self):
        _nonempty(self.observation_id, "observation_id")
        _nonempty(self.static_image, "static_image")
        _nonempty(self.wrist_image, "wrist_image")
        if self.environment_revision < 0 or self.captured_at_step < 0:
            raise ValueError("observation revision and step must be non-negative")


@dataclass(frozen=True)
class GoalInput:
    goal_id: str
    instruction: str
    success_spec: Dict[str, Any]
    initial_observation: ObservationRef
    target_reference_image: Optional[str] = None
    limits: RuntimeLimits = field(default_factory=RuntimeLimits)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self):
        _nonempty(self.goal_id, "goal_id")
        _nonempty(self.instruction, "instruction")
        if not isinstance(self.success_spec, dict) or not self.success_spec:
            raise ValueError("success_spec must be a non-empty mapping")
        if self.target_reference_image is not None:
            _nonempty(self.target_reference_image, "target_reference_image")
        self.initial_observation.validate()
        self.limits.validate()


@dataclass(frozen=True)
class GroundedObject:
    grounding_id: str
    role: Role
    source: str
    observation_id: str
    environment_revision: int
    representation: GroundingRepresentation
    semantic_id: Optional[str] = None
    camera: Optional[str] = None
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
    point_xy: Optional[Tuple[float, float]] = None
    mask_handle: Optional[str] = None
    tracker_handle: Optional[str] = None
    world_position: Optional[Tuple[float, float, float]] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self):
        _nonempty(self.grounding_id, "grounding_id")
        _nonempty(self.source, "grounding.source")
        _nonempty(self.observation_id, "grounding.observation_id")
        if self.environment_revision < 0:
            raise ValueError("grounding.environment_revision must be non-negative")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("grounding.confidence must be in [0, 1]")
        if self.representation == GroundingRepresentation.SEMANTIC:
            _nonempty(self.semantic_id, "grounding.semantic_id")
        elif self.representation == GroundingRepresentation.BBOX:
            _nonempty(self.camera, "grounding.camera")
            if self.bbox_xyxy is None or len(self.bbox_xyxy) != 4:
                raise ValueError("bbox grounding requires four bbox_xyxy values")
            x1, y1, x2, y2 = self.bbox_xyxy
            if not x1 < x2 or not y1 < y2:
                raise ValueError("bbox_xyxy must satisfy x1 < x2 and y1 < y2")
        elif self.representation == GroundingRepresentation.POINT:
            _nonempty(self.camera, "grounding.camera")
            if self.point_xy is None or len(self.point_xy) != 2:
                raise ValueError("point grounding requires two point_xy values")
        elif self.representation == GroundingRepresentation.MASK:
            _nonempty(self.mask_handle, "grounding.mask_handle")


@dataclass(frozen=True)
class ToolError:
    code: ErrorCode
    message: str
    retryable: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BudgetSnapshot:
    tool_calls_used: int
    tool_calls_remaining: int
    environment_steps_used: int
    environment_steps_remaining: int
    replans_used: int
    replans_remaining: int


@dataclass(frozen=True)
class ToolResultEnvelope:
    tool_call_id: str
    session_id: str
    status: ToolStatus
    data: Dict[str, Any]
    budget: BudgetSnapshot
    error: Optional[ToolError] = None
    artifacts: Tuple[ArtifactRef, ...] = ()
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self):
        if self.status in {ToolStatus.REJECTED, ToolStatus.UNAVAILABLE} and self.error is None:
            raise ValueError("rejected/unavailable tool result requires error")
        if self.status == ToolStatus.SUCCEEDED and self.error is not None:
            raise ValueError("succeeded tool result cannot contain error")

    def to_dict(self):
        return _json_value(asdict(self))


def _nonempty(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(path))


def _json_value(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
