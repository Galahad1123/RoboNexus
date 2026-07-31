"""Backend-neutral task execution and recovery data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class VerificationStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class TargetRole(str, Enum):
    SOURCE = "source"
    DESTINATION = "destination"
    INTERACTION = "interaction"


class MotionStatus(str, Enum):
    REACHED = "reached"
    UNSAFE = "unsafe"
    UNREACHABLE = "unreachable"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(slots=True, frozen=True)
class TaskSpec:
    task_id: str
    instruction: str
    oracle_task: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VerificationResult:
    status: VerificationStatus
    reason: str
    confidence: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LocalizationRequest:
    task: TaskSpec
    role: TargetRole
    selector: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TargetEstimate:
    target_id: str
    role: TargetRole
    frame: str
    position: np.ndarray
    orientation: np.ndarray | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Pose:
    frame: str
    position: np.ndarray
    orientation: np.ndarray | None = None


@dataclass(slots=True)
class ApproachPlan:
    target_id: str
    waypoints: list[Pose]
    clearance_m: float
    position_tolerance_m: float
    preserve_gripper_state: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MotionResult:
    status: MotionStatus
    steps: int
    reason: str
    final_position_error_m: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AttemptResult:
    success: bool
    steps: int
    termination_reason: str
    observation: Any
    metadata: dict[str, Any] = field(default_factory=dict)
