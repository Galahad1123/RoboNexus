"""Shared wire types.

Coordinates use the original RGB image space. ``bbox_xyxy`` is half-open:
``[x_min, y_min, x_max, y_max)``. Masks are uint8 arrays with values 0 or 1
and exactly the same height and width as their source frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ObjectRole(str, Enum):
    SOURCE = "source"
    DESTINATION = "destination"
    TOOL = "tool"
    OBSTACLE = "obstacle"


class DestinationType(str, Enum):
    OBJECT = "object"
    REGION = "region"
    RELATION = "relation"


class PipelineState(str, Enum):
    IDLE = "idle"
    LOCALIZING = "localizing"
    SEGMENTING = "segmenting"
    TRACKING = "tracking"
    RELOCALIZING = "relocalizing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class FramePacket:
    episode_id: str
    frame_id: int
    timestamp_ns: int
    rgb: np.ndarray
    camera_name: str = "static"

    def validate(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError(f"rgb must have shape HxWx3, got {self.rgb.shape}")
        if self.rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8, got {self.rgb.dtype}")
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")


@dataclass(slots=True)
class ObjectPrompt:
    role: ObjectRole
    label: str
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    destination_type: DestinationType | None = None
    relation: str | None = None

    def validate(self, width: int, height: int) -> None:
        x1, y1, x2, y2 = self.bbox_xyxy
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(
                f"invalid bbox {self.bbox_xyxy} for image {width}x{height}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")


@dataclass(slots=True)
class GroundingResult:
    action: str
    objects: list[ObjectPrompt]
    frame_id: int
    raw_response: str | None = None


@dataclass(slots=True)
class TrackedObject:
    object_id: int
    role: ObjectRole
    label: str
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray
    confidence: float
    quality: float = 1.0
    quality_signals: dict[str, float] = field(default_factory=dict)

    def validate(self, width: int, height: int) -> None:
        if self.mask.shape != (height, width):
            raise ValueError(
                f"mask shape {self.mask.shape} != frame shape {(height, width)}"
            )
        if self.mask.dtype != np.uint8:
            raise ValueError("mask must be uint8")


@dataclass(slots=True)
class PipelineResult:
    episode_id: str
    frame_id: int
    state: PipelineState
    action: str
    objects: list[TrackedObject]
    relocalized: bool = False
    error: str | None = None


def public_dict(value: Any) -> dict[str, Any]:
    """Convert dataclasses/enums to a JSON-friendly dict, excluding ndarray data."""
    result = asdict(value)
    for obj in result.get("objects", []):
        obj.pop("mask", None)
        for key, item in list(obj.items()):
            if isinstance(item, Enum):
                obj[key] = item.value
    for key, item in list(result.items()):
        if isinstance(item, Enum):
            result[key] = item.value
    return result
