"""Deterministic Qwen -> SAM -> Cutie orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .interfaces import Grounder, Segmenter, Tracker
from .quality import QualityConfig, evaluate_mask, mask_bbox
from ..common.schemas import (
    FramePacket,
    ObjectPrompt,
    PipelineResult,
    PipelineState,
    TrackedObject,
)


@dataclass(slots=True)
class PipelineConfig:
    quality: QualityConfig = field(default_factory=QualityConfig)
    sam_first_recovery: bool = True
    max_relocalizations: int = 3


class PerceptionPipeline:
    def __init__(
        self,
        grounder: Grounder,
        segmenter: Segmenter,
        tracker: Tracker,
        config: PipelineConfig | None = None,
    ) -> None:
        self.grounder = grounder
        self.segmenter = segmenter
        self.tracker = tracker
        self.config = config or PipelineConfig()
        self.state = PipelineState.IDLE
        self.instruction = ""
        self.action = ""
        self.session_id = ""
        self.prompts: dict[int, ObjectPrompt] = {}
        self.previous_masks: dict[int, np.ndarray] = {}
        self.last_good_masks: dict[int, np.ndarray] = {}
        self.failure_counts: dict[int, int] = {}
        self.relocalizations = 0
        self.tracker_session_active = False
        self.last_frame_id = -1

    def start(self, instruction: str, frame: FramePacket) -> PipelineResult:
        frame.validate()
        if self.state not in {PipelineState.IDLE, PipelineState.COMPLETED, PipelineState.FAILED}:
            raise RuntimeError(f"pipeline is already active in state {self.state.value}")
        self.instruction = instruction
        self.session_id = frame.episode_id
        self.state = PipelineState.LOCALIZING
        grounding = self.grounder.locate(instruction, frame.rgb, frame.frame_id)
        if not grounding.objects:
            return self._fail(frame, "grounder returned no objects")
        height, width = frame.rgb.shape[:2]
        for prompt in grounding.objects:
            prompt.validate(width, height)
        self.action = grounding.action
        self.state = PipelineState.SEGMENTING
        segmented = self.segmenter.segment(frame.rgb, grounding.objects, frame.frame_id)
        if len(segmented) != len(grounding.objects):
            return self._fail(frame, "segmenter response count does not match prompts")
        masks: dict[int, np.ndarray] = {}
        self.prompts.clear()
        for object_id, (prompt, (mask, score)) in enumerate(
            zip(grounding.objects, segmented), start=1
        ):
            self._validate_mask(mask, frame)
            if not mask.any():
                return self._fail(frame, f"empty initial mask for {prompt.label}")
            masks[object_id] = mask
            self.prompts[object_id] = prompt
        self.tracker.create_session(self.session_id)
        self.tracker_session_active = True
        tracked = self.tracker.initialize(
            self.session_id, frame.rgb, masks, frame.frame_id
        )
        self.previous_masks = {key: value[0].copy() for key, value in tracked.items()}
        self.last_good_masks = {key: value[0].copy() for key, value in tracked.items()}
        self.failure_counts = {key: 0 for key in tracked}
        self.relocalizations = 0
        self.last_frame_id = frame.frame_id
        self.state = PipelineState.TRACKING
        return self._result(frame, tracked, relocalized=False)

    def process(self, frame: FramePacket) -> PipelineResult:
        frame.validate()
        if self.state != PipelineState.TRACKING:
            raise RuntimeError(f"cannot process frame in state {self.state.value}")
        if frame.episode_id != self.session_id:
            raise ValueError("frame episode_id does not match active session")
        if frame.frame_id <= self.last_frame_id:
            raise ValueError(
                f"frame_id must increase monotonically: {frame.frame_id} <= {self.last_frame_id}"
            )
        tracked = self.tracker.track(self.session_id, frame.rgb, frame.frame_id)
        low_quality: list[int] = []
        for object_id, (mask, confidence) in tracked.items():
            self._validate_mask(mask, frame)
            score, _ = evaluate_mask(
                mask,
                self.previous_masks.get(object_id),
                confidence,
                self.config.quality,
            )
            self.failure_counts[object_id] = (
                self.failure_counts.get(object_id, 0) + 1
                if score < self.config.quality.min_score
                else 0
            )
            if self.failure_counts[object_id] >= self.config.quality.consecutive_failures:
                low_quality.append(object_id)
            elif score >= self.config.quality.min_score:
                self.last_good_masks[object_id] = mask.copy()
        relocalized = False
        if low_quality:
            tracked = self._recover(frame, low_quality)
            relocalized = True
        self.previous_masks = {key: value[0].copy() for key, value in tracked.items()}
        self.last_frame_id = frame.frame_id
        return self._result(frame, tracked, relocalized)

    def close(self, completed: bool = True) -> None:
        if self.session_id and self.tracker_session_active:
            self.tracker.delete_session(self.session_id)
            self.tracker_session_active = False
        self.state = PipelineState.COMPLETED if completed else PipelineState.FAILED

    def _recover(
        self, frame: FramePacket, object_ids: list[int]
    ) -> dict[int, tuple[np.ndarray, float]]:
        self.state = PipelineState.RELOCALIZING
        self.relocalizations += 1
        if self.relocalizations > self.config.max_relocalizations:
            self.state = PipelineState.FAILED
            raise RuntimeError("maximum relocalization count exceeded")

        prompts = [
            self._prompt_around_last_good(object_id, frame.rgb.shape[1], frame.rgb.shape[0])
            for object_id in object_ids
        ]
        if not self.config.sam_first_recovery:
            grounding = self.grounder.locate(
                self.instruction, frame.rgb, frame.frame_id
            )
            by_role = {item.role: item for item in grounding.objects}
            prompts = [by_role.get(item.role, item) for item in prompts]
        corrected = self.segmenter.segment(frame.rgb, prompts, frame.frame_id)
        masks = {
            object_id: result[0]
            for object_id, result in zip(object_ids, corrected)
            if result[0].any()
        }
        if len(masks) != len(object_ids) and self.config.sam_first_recovery:
            grounding = self.grounder.locate(
                self.instruction, frame.rgb, frame.frame_id
            )
            by_role = {item.role: item for item in grounding.objects}
            prompts = [by_role.get(item.role, item) for item in prompts]
            corrected = self.segmenter.segment(frame.rgb, prompts, frame.frame_id)
            masks = {
                object_id: result[0]
                for object_id, result in zip(object_ids, corrected)
                if result[0].any()
            }
        if len(masks) != len(object_ids):
            self.state = PipelineState.FAILED
            raise RuntimeError("relocalization produced an empty mask")
        for object_id, prompt in zip(object_ids, prompts):
            self.prompts[object_id] = prompt
            self.failure_counts[object_id] = 0
            self.last_good_masks[object_id] = masks[object_id].copy()
        tracked = self.tracker.correct(
            self.session_id, frame.rgb, masks, frame.frame_id
        )
        self.state = PipelineState.TRACKING
        return tracked

    def _result(
        self,
        frame: FramePacket,
        tracked: dict[int, tuple[np.ndarray, float]],
        relocalized: bool,
    ) -> PipelineResult:
        objects: list[TrackedObject] = []
        for object_id, (mask, confidence) in sorted(tracked.items()):
            score, signals = evaluate_mask(
                mask,
                self.previous_masks.get(object_id),
                confidence,
                self.config.quality,
            )
            prompt = self.prompts[object_id]
            objects.append(
                TrackedObject(
                    object_id=object_id,
                    role=prompt.role,
                    label=prompt.label,
                    bbox_xyxy=mask_bbox(mask),
                    mask=mask,
                    confidence=confidence,
                    quality=score,
                    quality_signals=signals,
                )
            )
        return PipelineResult(
            episode_id=frame.episode_id,
            frame_id=frame.frame_id,
            state=self.state,
            action=self.action,
            objects=objects,
            relocalized=relocalized,
        )

    def _validate_mask(self, mask: np.ndarray, frame: FramePacket) -> None:
        if mask.dtype != np.uint8 or mask.shape != frame.rgb.shape[:2]:
            raise ValueError(
                f"mask must be uint8 {frame.rgb.shape[:2]}, got {mask.dtype} {mask.shape}"
            )

    def _prompt_around_last_good(
        self, object_id: int, width: int, height: int
    ) -> ObjectPrompt:
        prompt = self.prompts[object_id]
        mask = self.last_good_masks.get(object_id)
        if mask is None or not mask.any():
            return prompt
        x1, y1, x2, y2 = mask_bbox(mask)
        margin_x = max(2, (x2 - x1) // 5)
        margin_y = max(2, (y2 - y1) // 5)
        return ObjectPrompt(
            role=prompt.role,
            label=prompt.label,
            bbox_xyxy=(
                max(0, x1 - margin_x),
                max(0, y1 - margin_y),
                min(width, x2 + margin_x),
                min(height, y2 + margin_y),
            ),
            confidence=prompt.confidence,
            destination_type=prompt.destination_type,
            relation=prompt.relation,
        )

    def _fail(self, frame: FramePacket, error: str) -> PipelineResult:
        self.state = PipelineState.FAILED
        return PipelineResult(
            episode_id=frame.episode_id,
            frame_id=frame.frame_id,
            state=self.state,
            action=self.action,
            objects=[],
            error=error,
        )
