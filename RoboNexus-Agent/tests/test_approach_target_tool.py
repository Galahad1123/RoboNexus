import unittest

import numpy as np

from backends.calvin.grounding.semantic_vocabulary import load_semantic_vocabulary
from backends.calvin.recovery.target_localizer import CalvinOracleTargetLocalizer
from core.domain import (
    ApproachExecutionResult,
    ArtifactRef,
    ErrorCode,
    GoalInput,
    GroundedObject,
    GroundingRepresentation,
    ObservationRef,
    Role,
    RuntimeLimits,
    SceneCapture,
    ToolStatus,
)
from core.ports import (
    ApproachControllerPort,
    ApproachRejected,
    ApproachUnavailable,
)
from core.runtime import AgentSession, SessionRejected
from core.tools import ApproachTargetTool


def make_session(environment_step_limit=100):
    initial = ObservationRef(
        "obs_000001", 0, 0, "artifact://initial/static.png", "artifact://initial/wrist.png"
    )
    return AgentSession(
        GoalInput(
            "goal-approach",
            "move near the red block",
            {"predicates": ["near_block_red"]},
            initial,
            limits=RuntimeLimits(environment_step_limit=environment_step_limit),
        ),
        session_id="session-approach",
    )


def add_grounding(session, semantic_id="block_red", world_position=None):
    grounding = GroundedObject(
        session.next_grounding_id(),
        Role.SOURCE,
        "calvin_gt",
        session.latest_observation.observation_id,
        session.environment_revision,
        GroundingRepresentation.SEMANTIC,
        semantic_id=semantic_id,
        camera="static",
        mask_handle="mask://session/target.npy",
        world_position=world_position,
        confidence=1.0,
        metadata={"instance_id": "movable:{}".format(semantic_id)},
    )
    session.add_grounding(grounding)
    return grounding


def params(grounding, threshold=None):
    result = {
        "grounding_id": grounding.grounding_id,
        "intent": "get closer before retrying the manipulation",
    }
    if threshold is not None:
        result["xy_threshold_m"] = threshold
    return result


class FakeArtifacts:
    def __init__(self):
        self.images = []

    def save_image(self, session_id, kind, image):
        self.images.append((kind, np.asarray(image).copy()))
        return ArtifactRef(kind, "artifact://{}/{}.png".format(session_id, kind), "image/png")


class FakeController(ApproachControllerPort):
    def __init__(self, success=True, steps=3, error=None):
        self.success = success
        self.steps = steps
        self.error = error
        self.calls = []

    def approach(self, grounding, xy_threshold_m, step_limit):
        self.calls.append((grounding, xy_threshold_m, step_limit))
        if self.error:
            raise self.error
        return ApproachExecutionResult(
            self.success,
            "reached_xy_threshold" if self.success else "motion_timeout",
            self.steps,
            0.30,
            0.06 if self.success else 0.22,
            xy_threshold_m,
            SceneCapture(
                np.zeros((6, 8, 3), dtype=np.uint8),
                np.zeros((4, 5, 3), dtype=np.uint8),
                self.steps,
                {"gripper": "open"},
                {},
            ),
        )


class ApproachTargetToolTest(unittest.TestCase):
    def test_success_updates_observation_and_invalidates_old_grounding(self):
        session = make_session()
        grounding = add_grounding(session)
        artifacts = FakeArtifacts()
        result = ApproachTargetTool(
            session, FakeController(), artifacts
        ).call(params(grounding))
        self.assertEqual(result.status, ToolStatus.SUCCEEDED)
        self.assertEqual(result.data["steps"], 3)
        self.assertEqual(result.data["threshold_m"], 0.08)
        self.assertEqual(result.data["after_observation_id"], "obs_000002")
        self.assertEqual(session.environment_steps_used, 3)
        self.assertEqual(session.environment_revision, 1)
        self.assertEqual(len(artifacts.images), 2)
        with self.assertRaises(SessionRejected) as raised:
            session.get_grounding(grounding.grounding_id)
        self.assertEqual(raised.exception.error.code, ErrorCode.STALE_GROUNDING)

    def test_zero_step_success_keeps_grounding_current(self):
        session = make_session()
        grounding = add_grounding(session)
        controller = FakeController(steps=0)
        result = ApproachTargetTool(
            session, controller, FakeArtifacts()
        ).call(params(grounding, 0.1))
        self.assertEqual(result.status, ToolStatus.SUCCEEDED)
        self.assertEqual(session.environment_revision, 0)
        self.assertIs(session.get_grounding(grounding.grounding_id), grounding)

    def test_motion_timeout_is_normal_failed_result(self):
        session = make_session()
        grounding = add_grounding(session)
        result = ApproachTargetTool(
            session, FakeController(success=False), FakeArtifacts()
        ).call(params(grounding))
        self.assertEqual(result.status, ToolStatus.FAILED)
        self.assertIsNone(result.error)
        self.assertFalse(result.data["success"])

    def test_threshold_is_bounded_and_boolean_is_rejected(self):
        for value in (0.019, 0.201, True, float("nan")):
            session = make_session()
            grounding = add_grounding(session)
            controller = FakeController()
            result = ApproachTargetTool(
                session, controller, FakeArtifacts()
            ).call(params(grounding, value))
            self.assertEqual(result.error.code, ErrorCode.INVALID_ARGUMENT)
            self.assertEqual(controller.calls, [])

    def test_remaining_environment_budget_is_controller_step_limit(self):
        session = make_session(environment_step_limit=5)
        grounding = add_grounding(session)
        controller = FakeController(steps=2)
        result = ApproachTargetTool(
            session, controller, FakeArtifacts()
        ).call(params(grounding))
        self.assertEqual(result.status, ToolStatus.SUCCEEDED)
        self.assertEqual(controller.calls[0][2], 5)

    def test_stale_grounding_is_rejected_before_motion(self):
        session = make_session()
        grounding = add_grounding(session)
        session.add_environment_steps(1)
        controller = FakeController()
        result = ApproachTargetTool(
            session, controller, FakeArtifacts()
        ).call(params(grounding))
        self.assertEqual(result.error.code, ErrorCode.STALE_GROUNDING)
        self.assertEqual(controller.calls, [])

    def test_backend_failures_are_structured(self):
        cases = (
            (ApproachRejected("no world target"), ToolStatus.REJECTED, ErrorCode.UNSUPPORTED_GROUNDING),
            (ApproachUnavailable("robot offline"), ToolStatus.UNAVAILABLE, ErrorCode.ENVIRONMENT_UNAVAILABLE),
        )
        for error, status, code in cases:
            session = make_session()
            grounding = add_grounding(session)
            result = ApproachTargetTool(
                session, FakeController(error=error), FakeArtifacts()
            ).call(params(grounding))
            self.assertEqual(result.status, status)
            self.assertEqual(result.error.code, code)

    def test_idempotent_replay_does_not_move_twice(self):
        session = make_session()
        grounding = add_grounding(session)
        controller = FakeController(steps=0)
        tool = ApproachTargetTool(session, controller, FakeArtifacts())
        first = tool.call(params(grounding), "same-call")
        second = tool.call(params(grounding, 0.15), "same-call")
        self.assertIs(first, second)
        self.assertEqual(len(controller.calls), 1)


class FakePhysics:
    def getBasePositionAndOrientation(self, uid, physicsClientId):
        return ([0.4, -0.1, 0.5], [0, 0, 0, 1])

    def getLinkState(self, uid, link, physicsClientId):
        return ([0.2, 0.3, 0.6], [0, 0, 0, 1])


class CalvinGroundingLocalizerTest(unittest.TestCase):
    def test_semantic_grounding_resolves_current_object_pose(self):
        scene_info = {
            "movable_objects": {"block_red": {"uid": 7}},
            "fixed_objects": {
                "table": {"uid": 9, "links": {"drawer_link": 3}}
            },
        }
        env = type(
            "Env",
            (),
            {
                "p": FakePhysics(),
                "cid": 0,
                "get_info": lambda self: {"scene_info": scene_info},
            },
        )()
        localizer = CalvinOracleTargetLocalizer(env)
        session = make_session()
        movable = add_grounding(session, "block_red")
        target_id, position, _ = localizer.locate_grounding(
            movable, load_semantic_vocabulary()
        )
        self.assertEqual(target_id, "block_red")
        np.testing.assert_allclose(position, [0.4, -0.1, 0.5])

    def test_explicit_world_position_does_not_query_scene(self):
        localizer = CalvinOracleTargetLocalizer(None)
        grounding = add_grounding(
            make_session(), "custom", world_position=(0.1, 0.2, 0.3)
        )
        _, position, orientation = localizer.locate_grounding(
            grounding, load_semantic_vocabulary()
        )
        np.testing.assert_allclose(position, [0.1, 0.2, 0.3])
        self.assertIsNone(orientation)


if __name__ == "__main__":
    unittest.main()
