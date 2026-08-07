import unittest
from pathlib import Path

import numpy as np

from backends.calvin.execution.outline_styles import load_outline_styles
from backends.calvin.execution.skill_executor import CalvinSkillExecutor
from backends.calvin.grounding.semantic_vocabulary import load_semantic_vocabulary
from core.domain import (
    ArtifactRef,
    ErrorCode,
    GoalInput,
    GroundedObject,
    GroundingRepresentation,
    ObservationRef,
    Role,
    RuntimeLimits,
    SceneCapture,
    SkillExecutionResult,
    ToolStatus,
)
from core.ports import (
    SkillExecutorPort,
    SkillExecutorUnavailable,
    SkillPreconditionRejected,
)
from core.runtime import AgentSession
from core.skills import load_skill_registry
from core.tools import ExecuteSkillTool


WORKSPACE = Path(__file__).resolve().parents[2]
SKILLS_CONFIG = WORKSPACE / "RoboNexus-Agent/configs/calvin/skills_d34.yaml"


def make_session(attempt_limit=3, environment_step_limit=2000):
    initial = ObservationRef(
        "obs_000001", 0, 0, "artifact://initial/static.png", "artifact://initial/wrist.png"
    )
    return AgentSession(
        GoalInput(
            "goal-execute",
            "lift the red block",
            {"predicates": ["lift_red_block_table"]},
            initial,
            limits=RuntimeLimits(
                skill_attempt_limit=attempt_limit,
                environment_step_limit=environment_step_limit,
            ),
        ),
        session_id="session-execute",
    )


def add_grounding(session, role="source", semantic_id="block_red", instance_id=None):
    grounding = GroundedObject(
        session.next_grounding_id(),
        Role(role),
        "calvin_gt",
        session.latest_observation.observation_id,
        session.environment_revision,
        GroundingRepresentation.SEMANTIC,
        semantic_id=semantic_id,
        camera="static",
        mask_handle="mask://session/{}.npy".format(semantic_id),
        confidence=1.0,
        metadata={"instance_id": instance_id or "movable:{}".format(semantic_id)},
    )
    session.add_grounding(grounding)
    return grounding


def params(skill_id, groundings):
    return {
        "skill_id": skill_id,
        "grounding_ids": [item.grounding_id for item in groundings],
        "intent": "advance the current atomic manipulation",
    }


class FakeArtifacts:
    def __init__(self):
        self.images = []
        self.videos = []

    def save_image(self, session_id, kind, image):
        self.images.append((kind, np.asarray(image).copy()))
        return ArtifactRef(kind, "artifact://{}/{}.png".format(session_id, kind), "image/png")

    def save_video(self, session_id, kind, frames, fps):
        self.videos.append((kind, tuple(frames), fps))
        return ArtifactRef(kind, "artifact://{}/{}.mp4".format(session_id, kind), "video/mp4")


class FakeExecutor(SkillExecutorPort):
    def __init__(self, success=True, steps=2, unavailable=False, precondition=None):
        self.success = success
        self.steps = steps
        self.unavailable = unavailable
        self.precondition = precondition
        self.checks = 0
        self.calls = []

    def check_preconditions(self, skill_id):
        if self.precondition:
            raise SkillPreconditionRejected(
                self.precondition, {"skill_id": skill_id}
            )
        return {"status": "ready"}

    def check_available(self, skill_id):
        self.checks += 1
        if self.unavailable:
            raise SkillExecutorUnavailable("endpoint offline")
        return "general"

    def execute(self, skill_id, groundings, attempt, logical_subtask_id, step_limit):
        self.calls.append(
            (skill_id, groundings, attempt, logical_subtask_id, step_limit)
        )
        capture = SceneCapture(
            np.full((6, 8, 3), 10, dtype=np.uint8),
            np.full((4, 5, 3), 20, dtype=np.uint8),
            self.steps,
            {"gripper": "closed", "holding_object": "unknown"},
            {"drawer": "closed"},
        )
        return SkillExecutionResult(
            "general",
            self.success,
            "success" if self.success else "timeout",
            self.steps,
            {
                "backend": "calvin_oracle",
                "status": "success" if self.success else "failure",
                "reason": "predicate_satisfied" if self.success else "predicate_not_satisfied",
            },
            capture,
            {"gripper": "closed", "holding_object": "unknown"},
            ("verify_goal",) if self.success else ("inspect_scene", "replan"),
            (np.zeros((6, 8, 3), dtype=np.uint8),),
            30,
        )


class ExecuteSkillToolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_skill_registry(SKILLS_CONFIG)

    def test_success_updates_scene_and_saves_video(self):
        session = make_session()
        source = add_grounding(session)
        artifacts = FakeArtifacts()
        result = ExecuteSkillTool(
            session, self.registry, FakeExecutor(success=True), artifacts
        ).call(params("lift_red_block_table", [source]))
        self.assertEqual(result.status, ToolStatus.SUCCEEDED)
        self.assertTrue(result.data["success"])
        self.assertEqual(result.data["attempt"], 1)
        self.assertEqual(result.data["policy_id"], "general")
        self.assertEqual(result.data["before_observation_id"], "obs_000001")
        self.assertEqual(result.data["after_observation_id"], "obs_000002")
        self.assertTrue(result.data["scene_continuation"])
        self.assertEqual(session.environment_steps_used, 2)
        self.assertEqual(session.environment_revision, 1)
        self.assertEqual(len(artifacts.images), 2)
        self.assertEqual(len(artifacts.videos), 1)

    def test_normal_policy_failure_is_failed_not_unavailable(self):
        session = make_session()
        source = add_grounding(session)
        result = ExecuteSkillTool(
            session, self.registry, FakeExecutor(success=False), FakeArtifacts()
        ).call(params("lift_red_block_table", [source]))
        self.assertEqual(result.status, ToolStatus.FAILED)
        self.assertIsNone(result.error)
        self.assertFalse(result.data["success"])
        self.assertEqual(result.data["verification"]["status"], "failure")

    def test_role_contract_is_enforced_before_endpoint_check(self):
        session = make_session()
        destination = add_grounding(session, "destination", "drawer")
        executor = FakeExecutor()
        result = ExecuteSkillTool(
            session, self.registry, executor, FakeArtifacts()
        ).call(params("lift_red_block_table", [destination]))
        self.assertEqual(result.status, ToolStatus.REJECTED)
        self.assertEqual(result.error.code, ErrorCode.INVALID_PRECONDITION)
        self.assertEqual(executor.checks, 0)

    def test_source_and_destination_must_be_distinct(self):
        session = make_session()
        source = add_grounding(session, "source", "block_red", "same-instance")
        destination = add_grounding(
            session, "destination", "block_red", "same-instance"
        )
        result = ExecuteSkillTool(
            session, self.registry, FakeExecutor(), FakeArtifacts()
        ).call(params("stack_block", [source, destination]))
        self.assertEqual(result.error.code, ErrorCode.INVALID_PRECONDITION)

    def test_unavailable_endpoint_does_not_consume_attempt(self):
        session = make_session()
        source = add_grounding(session)
        result = ExecuteSkillTool(
            session,
            self.registry,
            FakeExecutor(unavailable=True),
            FakeArtifacts(),
        ).call(params("lift_red_block_table", [source]))
        self.assertEqual(result.status, ToolStatus.UNAVAILABLE)
        self.assertEqual(result.error.code, ErrorCode.POLICY_UNAVAILABLE)
        logical_id = session.resolve_logical_subtask("lift_red_block_table", (source,))
        self.assertEqual(session.attempts_used(logical_id), 0)

    def test_invalid_precondition_does_not_contact_policy_or_consume_attempt(self):
        session = make_session()
        destination = add_grounding(session, "destination", "drawer")
        executor = FakeExecutor(precondition="drawer_must_be_open")
        result = ExecuteSkillTool(
            session, self.registry, executor, FakeArtifacts()
        ).call(params("place_in_drawer", [destination]))
        self.assertEqual(result.status, ToolStatus.REJECTED)
        self.assertEqual(result.error.code, ErrorCode.INVALID_PRECONDITION)
        self.assertEqual(result.error.message, "drawer_must_be_open")
        self.assertEqual(executor.checks, 0)
        logical_id = session.resolve_logical_subtask("place_in_drawer", (destination,))
        self.assertEqual(session.attempts_used(logical_id), 0)

    def test_attempt_budget_is_shared_after_regrounding(self):
        session = make_session(attempt_limit=2)
        executor = FakeExecutor(success=False, steps=0)
        tool = ExecuteSkillTool(session, self.registry, executor, FakeArtifacts())
        first = add_grounding(session)
        result1 = tool.call(params("lift_red_block_table", [first]), "attempt-1")
        second = add_grounding(session)
        result2 = tool.call(params("lift_red_block_table", [second]), "attempt-2")
        third = add_grounding(session)
        result3 = tool.call(params("lift_red_block_table", [third]), "attempt-3")
        self.assertEqual(result1.data["logical_subtask_id"], result2.data["logical_subtask_id"])
        self.assertEqual((result1.data["attempt"], result2.data["attempt"]), (1, 2))
        self.assertEqual(result3.error.code, ErrorCode.ATTEMPT_BUDGET_EXHAUSTED)
        self.assertEqual(len(executor.calls), 2)

    def test_idempotent_replay_does_not_execute_twice(self):
        session = make_session()
        source = add_grounding(session)
        executor = FakeExecutor(steps=0)
        tool = ExecuteSkillTool(session, self.registry, executor, FakeArtifacts())
        first = tool.call(params("lift_red_block_table", [source]), "same-call")
        second = tool.call(params("lift_red_block_table", [source]), "same-call")
        self.assertIs(first, second)
        self.assertEqual(len(executor.calls), 1)

    def test_runtime_caps_attempt_to_remaining_environment_budget(self):
        session = make_session(environment_step_limit=7)
        source = add_grounding(session)
        executor = FakeExecutor(steps=2)
        result = ExecuteSkillTool(
            session, self.registry, executor, FakeArtifacts()
        ).call(params("lift_red_block_table", [source]))
        self.assertEqual(result.status, ToolStatus.SUCCEEDED)
        self.assertEqual(executor.calls[0][4], 7)


class CalvinSkillExecutorConfigTest(unittest.TestCase):
    def test_outline_styles_cover_all_policy_groups(self):
        styles = load_outline_styles()
        self.assertEqual(set(styles), {"general", "push", "rotate", "stack"})

    def test_groundings_convert_to_dynamic_oracle_selectors(self):
        executor = object.__new__(CalvinSkillExecutor)
        executor.vocabulary = load_semantic_vocabulary()
        session = make_session()
        source = add_grounding(session)
        destination = add_grounding(session, "destination", "drawer")
        roles = executor._mask_roles((source, destination))
        self.assertEqual(roles["source"], [{"movable": "block_red"}])
        self.assertEqual(
            roles["destination"],
            [{"fixed": "table", "links": ["drawer_link"]}],
        )


if __name__ == "__main__":
    unittest.main()
