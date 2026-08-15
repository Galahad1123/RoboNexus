"""Tests for the active Agent session runtime."""

import unittest

from core.domain import (
    AgentSessionStatus,
    ErrorCode,
    GoalInput,
    GroundedObject,
    GroundingRepresentation,
    ObservationRef,
    Role,
    RuntimeLimits,
    ToolStatus,
)
from core.runtime import AgentSession, SessionRejected


def make_session(**limit_overrides):
    limits = RuntimeLimits(**limit_overrides)
    observation = ObservationRef(
        "obs_000001",
        environment_revision=0,
        captured_at_step=0,
        static_image="artifact://test/static.png",
        wrist_image="artifact://test/wrist.png",
    )
    goal = GoalInput(
        goal_id="goal-1",
        instruction="put the red block in the drawer",
        success_spec={"predicates": ["place_in_drawer"]},
        initial_observation=observation,
        target_reference_image="artifact://test/goal.png",
        limits=limits,
    )
    return AgentSession(goal, session_id="session-1")


class AgentSessionRuntimeTest(unittest.TestCase):
    def test_recovery_success_does_not_advance_private_goal_sequence(self):
        session = make_session()
        session.goal.success_spec["mode"] = "skill_sequence"
        session.goal.success_spec["predicates"] = [
            {"name": "lift_red_block_table"},
            {"name": "place_in_drawer"},
        ]
        self.assertEqual(session.classify_successful_skill("open_drawer"), "recovery")
        session.record_recovery_achievement("open_drawer", "subtask-r")
        self.assertEqual(session.achievement_sequence, ())
        self.assertEqual(session.classify_successful_skill("lift_red_block_table"), "goal_progress")
        session.record_skill_achievement("lift_red_block_table", "subtask-1")
        self.assertEqual(session.achievement_sequence, ("lift_red_block_table",))

    def test_goal_requires_success_spec(self):
        observation = ObservationRef("o", 0, 0, "static", "wrist")
        with self.assertRaisesRegex(ValueError, "success_spec"):
            GoalInput("g", "task", {}, observation).validate()

    def test_initial_budget_and_observation(self):
        session = make_session()
        self.assertEqual(session.status, AgentSessionStatus.ACTIVE)
        self.assertEqual(session.latest_observation.observation_id, "obs_000001")
        self.assertEqual(session.budget_snapshot().tool_calls_remaining, 32)

    def test_tool_call_replay_is_idempotent(self):
        session = make_session()
        call_id = session.next_tool_call_id()
        self.assertIsNone(session.start_tool_call(call_id, "inspect_scene"))
        result = session.result(call_id, ToolStatus.SUCCEEDED, {"ok": True})
        session.finish_tool_call(result)
        replay = session.start_tool_call(call_id, "inspect_scene")
        self.assertIs(replay, result)
        self.assertEqual(session.tool_calls_used, 1)

    def test_completed_terminal_tool_call_can_be_replayed(self):
        session = make_session()
        session.start_tool_call("terminal", "verify_goal")
        result = session.result("terminal", ToolStatus.SUCCEEDED, {"status": "success"})
        session.mark_succeeded("success", "verified")
        session.finish_tool_call(result)
        self.assertIs(session.start_tool_call("terminal", "verify_goal"), result)

    def test_duplicate_running_tool_call_is_rejected(self):
        session = make_session()
        session.start_tool_call("tc-x", "execute_skill")
        with self.assertRaises(SessionRejected) as caught:
            session.start_tool_call("tc-x", "execute_skill")
        self.assertEqual(caught.exception.error.code, ErrorCode.TOOL_CALL_IN_PROGRESS)

    def test_tool_call_budget_terminates_session(self):
        session = make_session(tool_call_limit=1)
        session.start_tool_call("tc-1", "inspect_scene")
        session.finish_tool_call(
            session.result("tc-1", ToolStatus.SUCCEEDED, {})
        )
        with self.assertRaises(SessionRejected) as caught:
            session.start_tool_call("tc-2", "inspect_scene")
        self.assertEqual(caught.exception.error.code, ErrorCode.GOAL_BUDGET_EXHAUSTED)
        self.assertEqual(session.status, AgentSessionStatus.BUDGET_EXHAUSTED)

    def test_observation_revision_changes_only_with_environment(self):
        session = make_session()
        read_only = session.add_observation("s2", "w2")
        self.assertEqual(read_only.environment_revision, 0)
        session.add_environment_steps(4)
        changed = session.add_observation("s3", "w3")
        self.assertEqual(changed.environment_revision, 1)
        self.assertEqual(changed.captured_at_step, 4)

    def test_old_observation_becomes_stale_for_action(self):
        session = make_session()
        session.add_environment_steps(1)
        with self.assertRaises(SessionRejected) as caught:
            session.get_observation("obs_000001", require_current=True)
        self.assertEqual(caught.exception.error.code, ErrorCode.STALE_OBSERVATION)

    def test_environment_step_budget_is_hard_limit(self):
        session = make_session(environment_step_limit=3)
        with self.assertRaises(SessionRejected):
            session.add_environment_steps(4)
        self.assertEqual(session.environment_steps_used, 0)
        self.assertEqual(session.status, AgentSessionStatus.BUDGET_EXHAUSTED)

    def test_attempt_budget_is_scoped_to_logical_subtask(self):
        session = make_session(skill_attempt_limit=2)
        self.assertEqual(session.consume_attempt("subtask-a"), 1)
        self.assertEqual(session.consume_attempt("subtask-a"), 2)
        with self.assertRaises(SessionRejected) as caught:
            session.consume_attempt("subtask-a")
        self.assertEqual(caught.exception.error.code, ErrorCode.ATTEMPT_BUDGET_EXHAUSTED)
        self.assertEqual(session.consume_attempt("subtask-b"), 1)

    def test_goal_stage_budget_refreshes_after_verified_progress(self):
        session = make_session()
        session.goal.success_spec.update(
            {
                "mode": "skill_sequence",
                "predicates": [
                    {"name": "open_drawer"},
                    {"name": "close_drawer"},
                ],
            }
        )
        first_scope = session.attempt_scope("open_drawer", "logical-a")
        self.assertEqual(first_scope, "goal_stage_01:open_drawer")
        self.assertEqual(session.consume_attempt(first_scope), 1)
        session.record_skill_achievement("open_drawer", "logical-a")
        second_scope = session.attempt_scope("close_drawer", "logical-b")
        self.assertEqual(second_scope, "goal_stage_02:close_drawer")
        self.assertEqual(session.attempts_used(second_scope), 0)

    def test_recovery_does_not_consume_future_goal_stage_budget(self):
        session = make_session()
        session.goal.success_spec.update(
            {
                "mode": "skill_sequence",
                "predicates": [
                    {"name": "open_drawer"},
                    {"name": "lift_red_block_table"},
                ],
            }
        )
        recovery_scope = session.attempt_scope(
            "lift_red_block_table", "logical-recovery"
        )
        self.assertTrue(recovery_scope.startswith("recovery:"))
        session.consume_attempt(recovery_scope)
        session.record_recovery_achievement(
            "lift_red_block_table", "logical-recovery"
        )
        session.record_skill_achievement("open_drawer", "logical-goal")
        goal_scope = session.attempt_scope(
            "lift_red_block_table", "logical-recovery"
        )
        self.assertEqual(goal_scope, "goal_stage_02:lift_red_block_table")
        self.assertEqual(session.attempts_used(goal_scope), 0)

    def test_replan_budget(self):
        session = make_session(replan_limit=1)
        self.assertEqual(session.consume_replan(), 1)
        with self.assertRaises(SessionRejected) as caught:
            session.consume_replan()
        self.assertEqual(caught.exception.error.code, ErrorCode.REPLAN_BUDGET_EXHAUSTED)

    def test_grounding_is_bound_to_environment_revision(self):
        session = make_session()
        grounding = GroundedObject(
            grounding_id=session.next_grounding_id(),
            role=Role.TARGET,
            source="calvin_gt",
            observation_id="obs_000001",
            environment_revision=0,
            representation=GroundingRepresentation.SEMANTIC,
            semantic_id="block_red",
            mask_handle="mask://red",
            confidence=1.0,
        )
        session.add_grounding(grounding)
        self.assertIs(session.get_grounding(grounding.grounding_id), grounding)
        session.add_environment_steps(1)
        with self.assertRaises(SessionRejected) as caught:
            session.get_grounding(grounding.grounding_id)
        self.assertEqual(caught.exception.error.code, ErrorCode.STALE_GROUNDING)

    def test_tracker_grounding_survives_revision(self):
        session = make_session()
        grounding = GroundedObject(
            "ground-tracked",
            Role.TARGET,
            "cutie",
            "obs_000001",
            0,
            GroundingRepresentation.MASK,
            mask_handle="mask://tracked",
            tracker_handle="tracker://one",
        )
        session.add_grounding(grounding)
        session.add_environment_steps(1)
        self.assertIs(session.get_grounding("ground-tracked"), grounding)

    def test_success_requires_verified_success(self):
        session = make_session()
        with self.assertRaises(SessionRejected):
            session.mark_succeeded("failure", "not done")
        session.mark_succeeded("success", "oracle confirmed")
        self.assertEqual(session.status, AgentSessionStatus.SUCCEEDED)
        with self.assertRaises(SessionRejected):
            session.add_observation("s", "w")

    def test_abort_is_explicit_terminal_transition(self):
        session = make_session()
        session.abort(AgentSessionStatus.ABORTED_BY_AGENT, "no viable plan")
        self.assertEqual(session.status, AgentSessionStatus.ABORTED_BY_AGENT)
        with self.assertRaises(SessionRejected):
            session.consume_attempt("anything")

    def test_envelope_serializes_enums(self):
        session = make_session()
        result = session.result("tc-1", ToolStatus.SUCCEEDED, {"value": 1})
        payload = result.to_dict()
        self.assertEqual(payload["protocol_version"], 1)
        self.assertEqual(payload["status"], "succeeded")
        self.assertEqual(payload["budget"]["tool_calls_remaining"], 32)


if __name__ == "__main__":
    unittest.main()
