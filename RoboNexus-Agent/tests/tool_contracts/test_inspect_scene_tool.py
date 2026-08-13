import tempfile
"""Tests for the inspect_scene tool contract."""

import unittest
from pathlib import Path

import numpy as np

from backends.calvin.environment.scene_observer import CalvinSceneObserver
from backends.local_artifact_store import LocalArtifactStore
from core.domain import (
    ArtifactRef,
    ErrorCode,
    GoalInput,
    ObservationRef,
    RuntimeLimits,
    SceneCapture,
    ToolStatus,
)
from core.ports import ArtifactStorePort, EnvironmentPort, EnvironmentPortError
from core.runtime import AgentSession
from core.tools import InspectSceneTool


def make_session(tool_call_limit=32, environment_step_limit=2000):
    initial = ObservationRef(
        "obs_000001", 0, 0, "artifact://initial/static.png", "artifact://initial/wrist.png"
    )
    return AgentSession(
        GoalInput(
            "goal-inspect",
            "inspect the scene",
            {"predicates": ["scene_inspected"]},
            initial,
            limits=RuntimeLimits(
                tool_call_limit=tool_call_limit,
                environment_step_limit=environment_step_limit,
            ),
        ),
        session_id="session-inspect",
    )


class FakeEnvironment(EnvironmentPort):
    def __init__(self, steps=0, error=None):
        self.steps = steps
        self.error = error
        self.calls = 0

    @property
    def environment_steps(self):
        return self.steps

    def capture_scene(self):
        self.calls += 1
        if self.error:
            raise self.error
        return SceneCapture(
            np.full((8, 10, 3), 10, dtype=np.uint8),
            np.full((6, 7, 3), 20, dtype=np.uint8),
            self.steps,
            {"gripper": "open", "holding_object": "unknown"},
            {"drawer": "closed", "slider": "left"},
        )


class FakeArtifacts(ArtifactStorePort):
    def __init__(self):
        self.images = []

    def save_image(self, session_id, kind, image):
        self.images.append((session_id, kind, np.asarray(image).copy()))
        return ArtifactRef(kind, "artifact://{}/{}.png".format(session_id, kind), "image/png")

    def save_video(self, session_id, kind, frames, fps):
        raise NotImplementedError


class InspectSceneToolTest(unittest.TestCase):
    def test_returns_two_images_and_state_summaries(self):
        session = make_session()
        environment = FakeEnvironment()
        artifacts = FakeArtifacts()
        result = InspectSceneTool(session, environment, artifacts).call(
            {"reason": "choose the first skill"}, tool_call_id="tc-inspect"
        )
        self.assertEqual(result.status, ToolStatus.SUCCEEDED)
        self.assertEqual(result.data["reason"], "choose the first skill")
        self.assertEqual(result.data["robot_summary"]["gripper"], "open")
        self.assertEqual(result.data["facility_summary"]["drawer"], "closed")
        self.assertEqual(len(result.artifacts), 2)
        self.assertEqual(len(artifacts.images), 2)
        self.assertEqual(session.latest_observation.observation_id, "obs_000002")
        self.assertEqual(session.environment_steps_used, 0)

    def test_summary_sections_can_be_omitted(self):
        result = InspectSceneTool(
            make_session(), FakeEnvironment(), FakeArtifacts()
        ).call(
            {
                "include_robot_summary": False,
                "include_facility_summary": False,
            }
        )
        self.assertNotIn("robot_summary", result.data)
        self.assertNotIn("facility_summary", result.data)

    def test_read_only_replay_does_not_capture_or_save_twice(self):
        session = make_session()
        environment = FakeEnvironment()
        artifacts = FakeArtifacts()
        tool = InspectSceneTool(session, environment, artifacts)
        first = tool.call({}, tool_call_id="same-call")
        second = tool.call({}, tool_call_id="same-call")
        self.assertIs(first, second)
        self.assertEqual(environment.calls, 1)
        self.assertEqual(len(artifacts.images), 2)
        self.assertEqual(session.tool_calls_used, 1)

    def test_invalid_params_are_counted_rejected_and_replayable(self):
        session = make_session()
        environment = FakeEnvironment()
        tool = InspectSceneTool(session, environment, FakeArtifacts())
        first = tool.call({"unexpected": 1}, tool_call_id="bad-call")
        second = tool.call({"unexpected": 2}, tool_call_id="bad-call")
        self.assertEqual(first.status, ToolStatus.REJECTED)
        self.assertEqual(first.error.code, ErrorCode.INVALID_ARGUMENT)
        self.assertIs(first, second)
        self.assertEqual(session.tool_calls_used, 1)
        self.assertEqual(environment.calls, 0)

    def test_external_environment_progress_updates_revision(self):
        session = make_session()
        result = InspectSceneTool(
            session, FakeEnvironment(steps=3), FakeArtifacts()
        ).call({})
        self.assertEqual(result.status, ToolStatus.SUCCEEDED)
        self.assertEqual(session.environment_steps_used, 3)
        self.assertEqual(session.environment_revision, 1)
        self.assertEqual(result.data["observation"]["captured_at_step"], 3)
        self.assertEqual(result.data["observation"]["environment_revision"], 1)

    def test_backward_environment_counter_is_unavailable(self):
        session = make_session()
        session.add_environment_steps(2)
        result = InspectSceneTool(
            session, FakeEnvironment(steps=1), FakeArtifacts()
        ).call({})
        self.assertEqual(result.status, ToolStatus.UNAVAILABLE)
        self.assertEqual(result.error.code, ErrorCode.ENVIRONMENT_UNAVAILABLE)

    def test_environment_failure_is_structured(self):
        result = InspectSceneTool(
            make_session(),
            FakeEnvironment(error=EnvironmentPortError("camera unavailable")),
            FakeArtifacts(),
        ).call({})
        self.assertEqual(result.status, ToolStatus.UNAVAILABLE)
        self.assertEqual(result.error.code, ErrorCode.ENVIRONMENT_UNAVAILABLE)
        self.assertTrue(result.error.retryable)

    def test_local_artifact_store_writes_resolvable_pngs(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            result = InspectSceneTool(
                make_session(), FakeEnvironment(), store
            ).call({})
            for artifact in result.artifacts:
                path = store.resolve(artifact.uri)
                self.assertTrue(path.is_file())
                self.assertTrue(str(path).startswith(str(Path(directory).resolve())))

    def test_calvin_scene_observer_copies_both_views(self):
        scene = np.zeros(24, dtype=np.float32)
        scene[0] = 0.25
        scene[1] = 0.0
        scene[4] = 1.0
        robot = np.zeros(15, dtype=np.float32)
        robot[14] = 1.0
        adapter = type(
            "Adapter",
            (),
            {
                "environment_steps": 7,
                "current_observation": {
                    "rgb_obs": {
                        "rgb_static": np.full((5, 6, 3), 31, dtype=np.uint8),
                        "rgb_gripper": np.full((3, 4, 3), 47, dtype=np.uint8),
                    },
                    "robot_obs": robot,
                    "scene_obs": scene,
                },
            },
        )()
        capture = CalvinSceneObserver(adapter).capture_scene()
        self.assertEqual(capture.environment_steps, 7)
        self.assertEqual(capture.static_rgb.shape, (5, 6, 3))
        self.assertEqual(capture.wrist_rgb.shape, (3, 4, 3))
        self.assertEqual(capture.robot_summary["gripper"], "open")
        self.assertEqual(capture.facility_summary["drawer"], "closed")
        self.assertEqual(capture.facility_summary["slider"], "left")


if __name__ == "__main__":
    unittest.main()
