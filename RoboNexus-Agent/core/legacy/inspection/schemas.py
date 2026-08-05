"""Legacy state inspection results consumed by scripted execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class InspectionStatus(str, Enum):
    READY = "ready"
    SATISFIED = "satisfied"
    INVALID_PRECONDITION = "invalid_precondition"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SkillInspection:
    status: InspectionStatus
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
