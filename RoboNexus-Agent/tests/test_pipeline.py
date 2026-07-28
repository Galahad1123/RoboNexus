from __future__ import annotations

import unittest

import numpy as np

from core.orchestrator import PerceptionPipeline, PipelineConfig
from core.quality import QualityConfig, mask_iou
from core.schemas import FramePacket, GroundingResult, ObjectPrompt, ObjectRole, PipelineState
from segmentation_service.backend import MockSamBackend
from tracking_service.backend import MockCutieBackend


class Grounder:
    def __init__(self) -> None:
        self.calls = 0

    def locate(self, instruction, rgb, frame_id):
        self.calls += 1
        return GroundingResult(
            action="place",
            objects=[
                ObjectPrompt(ObjectRole.SOURCE, "red block", (2, 2, 8, 8), 0.9),
                ObjectPrompt(ObjectRole.DESTINATION, "drawer", (10, 2, 18, 10), 0.8),
            ],
            frame_id=frame_id,
        )


class Segmenter:
    def __init__(self) -> None:
        self.backend = MockSamBackend()

    def segment(self, rgb, prompts, frame_id):
        return self.backend.segment(
            rgb, [{"bbox_xyxy": prompt.bbox_xyxy} for prompt in prompts]
        )


class FailingTracker(MockCutieBackend):
    def __init__(self) -> None:
        super().__init__()
        self.track_calls = 0
        self.correct_calls = 0

    def track(self, session_id, rgb, frame_id=None):
        self.track_calls += 1
        if self.track_calls <= 2:
            return {
                key: (np.zeros_like(mask), 0.0)
                for key, mask in self.sessions[session_id].masks.items()
            }
        return super().track(session_id, rgb)

    def correct(self, session_id, rgb, masks, frame_id=None):
        self.correct_calls += 1
        return super().correct(session_id, rgb, masks, frame_id)


def frame(frame_id=0):
    return FramePacket("episode", frame_id, frame_id, np.zeros((20, 20, 3), np.uint8))


class PipelineTest(unittest.TestCase):
    def test_initialize_and_track(self):
        pipeline = PerceptionPipeline(Grounder(), Segmenter(), MockCutieBackend())
        first = pipeline.start("place red block in drawer", frame())
        self.assertEqual(first.state, PipelineState.TRACKING)
        self.assertEqual([item.role for item in first.objects], [ObjectRole.SOURCE, ObjectRole.DESTINATION])
        result = pipeline.process(frame(1))
        self.assertFalse(result.relocalized)
        self.assertEqual(mask_iou(first.objects[0].mask, result.objects[0].mask), 1.0)
        pipeline.close()
        self.assertEqual(pipeline.state, PipelineState.COMPLETED)

    def test_two_bad_frames_trigger_sam_correction(self):
        tracker = FailingTracker()
        pipeline = PerceptionPipeline(
            Grounder(),
            Segmenter(),
            tracker,
            PipelineConfig(
                quality=QualityConfig(min_score=0.8, consecutive_failures=2)
            ),
        )
        pipeline.start("place red block in drawer", frame())
        self.assertFalse(pipeline.process(frame(1)).relocalized)
        result = pipeline.process(frame(2))
        self.assertTrue(result.relocalized)
        self.assertEqual(tracker.correct_calls, 1)

    def test_rejects_wrong_episode(self):
        pipeline = PerceptionPipeline(Grounder(), Segmenter(), MockCutieBackend())
        pipeline.start("place", frame())
        packet = FramePacket("other", 1, 1, np.zeros((20, 20, 3), np.uint8))
        with self.assertRaises(ValueError):
            pipeline.process(packet)

    def test_rejects_out_of_order_frame(self):
        pipeline = PerceptionPipeline(Grounder(), Segmenter(), MockCutieBackend())
        pipeline.start("place", frame())
        with self.assertRaises(ValueError):
            pipeline.process(frame())


if __name__ == "__main__":
    unittest.main()
