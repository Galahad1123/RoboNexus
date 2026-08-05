"""Legacy retry-era task execution and recovery data contracts."""

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


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    instruction: str
    oracle_task: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    status: VerificationStatus
    reason: str
    confidence: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalizationRequest:
    task: TaskSpec
    role: TargetRole
    selector: dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetEstimate:
    target_id: str
    role: TargetRole
    frame: str
    position: np.ndarray
    orientation: np.ndarray | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Pose:
    frame: str
    position: np.ndarray
    orientation: np.ndarray | None = None


@dataclass
class ApproachPlan:
    target_id: str
    waypoints: list[Pose]
    clearance_m: float
    position_tolerance_m: float
    preserve_gripper_state: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionResult:
    status: MotionStatus
    steps: int
    reason: str
    final_position_error_m: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttemptResult:
    success: bool
    steps: int
    termination_reason: str
    observation: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AtomicAttemptContext:
    execution_id: str
    policy_episode_id: str
    skill_id: str
    policy_id: str
    instruction: str
    attempt: int
    episode_seed: int
    replan_interval: int


@dataclass
class PolicyStepResult:
    action: np.ndarray
    policy_id: str
    skill_id: str
    attempt: int
    step_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AtomicAttemptSummary:
    context: AtomicAttemptContext
    success: bool
    steps: int
    termination_reason: str
