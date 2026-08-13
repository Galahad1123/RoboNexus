"""Tests for verify_goal and abort_goal."""

import unittest

from backends.calvin.verification.goal_verifier import CalvinOracleGoalVerifier
from core.domain import (
    AgentSessionStatus,
    ErrorCode,
    GoalInput,
    GoalVerificationResult,
    ObservationRef,
    ToolStatus,
)
from core.ports import GoalVerifierPort, GoalVerifierUnavailable
from core.runtime import AgentSession
from core.tools import AbortGoalTool, VerifyGoalTool


def make_session(with_reference=True, predicates=None, mode=None):
    observation = ObservationRef(
        "obs_000001", 0, 0, "artifact://initial/static.png", "artifact://initial/wrist.png"
    )
    success_spec = {"predicates": predicates or ["place_in_drawer"]}
    if mode is not None:
        success_spec["mode"] = mode
    goal = GoalInput(
        "goal-terminal",
        "put the red block in the drawer",
        success_spec,
        observation,
        target_reference_image=(
            "artifact://goal/reference.png" if with_reference else None
        ),
    )
    return AgentSession(goal, session_id="session-terminal")


class FakeVerifier(GoalVerifierPort):
    def __init__(self, status="success", error=None):
        self.status = status
        self.error = error
        self.calls = []

    def verify(self, goal, observation, use_goal_reference=True):
        self.calls.append((goal, observation, use_goal_reference))
        if self.error:
            raise self.error
        return GoalVerificationResult(
            self.status,
            "fake_verifier",
            "fake_{}".format(self.status),
            0.9,
            {"predicates": [{"name": "place_in_drawer", "satisfied": self.status == "success"}]},
        )


class VerifyGoalToolTest(unittest.TestCase):
    def test_skill_sequence_latches_oracle_confirmed_intermediate_skills(self):
        session = make_session(
            predicates=["lift_red_block_table", "place_in_drawer"],
            mode="skill_sequence",
        )
        session.record_skill_achievement("lift_red_block_table", "subtask_001")
        session.record_skill_achievement("place_in_drawer", "subtask_002")
        result = VerifyGoalTool(session, FakeVerifier("failure")).call(
            {"claim": "sequence complete", "observation_id": "obs_000001"}
        )
        self.assertEqual(result.data["status"], "success")
        self.assertEqual(session.status, AgentSessionStatus.SUCCEEDED)
        self.assertEqual(
            result.data["evidence"]["achieved_skill_ids"],
            ["lift_red_block_table", "place_in_drawer"],
        )
        self.assertTrue(result.data["evidence"]["sequence_order_satisfied"])

    def test_skill_sequence_rejects_reverse_achievement_order(self):
        session = make_session(
            predicates=["open_drawer", "close_drawer"],
            mode="skill_sequence",
        )
        session.record_skill_achievement("close_drawer", "subtask_001")
        session.record_skill_achievement("open_drawer", "subtask_002")
        result = VerifyGoalTool(session, FakeVerifier("success")).call(
            {"claim": "sequence complete", "observation_id": "obs_000001"}
        )
        self.assertEqual(result.data["status"], "failure")
        self.assertFalse(result.data["evidence"]["sequence_order_satisfied"])
        self.assertEqual(session.status, AgentSessionStatus.ACTIVE)

    def test_success_is_the_only_verifier_result_that_terminates_successfully(self):
        session = make_session()
        tool = VerifyGoalTool(session, FakeVerifier("success"))
        first = tool.call(
            {
                "claim": "the block is in the drawer",
                "observation_id": "obs_000001",
            },
            "verify-once",
        )
        self.assertEqual(first.status, ToolStatus.SUCCEEDED)
        self.assertEqual(first.data["status"], "success")
        self.assertTrue(first.data["goal_reference_used"])
        self.assertEqual(session.status, AgentSessionStatus.SUCCEEDED)
        self.assertEqual(session.termination_reason, "fake_success")
        replay = tool.call({}, "verify-once")
        self.assertIs(replay, first)
        rejected = tool.call(
            {"claim": "again", "observation_id": "obs_000001"}, "new-call"
        )
        self.assertEqual(rejected.error.code, ErrorCode.SESSION_TERMINAL)

    def test_failure_and_uncertain_leave_session_active(self):
        for status in ("failure", "uncertain"):
            session = make_session()
            result = VerifyGoalTool(session, FakeVerifier(status)).call(
                {"claim": "done", "observation_id": "obs_000001"}
            )
            self.assertEqual(result.status, ToolStatus.SUCCEEDED)
            self.assertEqual(result.data["status"], status)
            self.assertEqual(session.status, AgentSessionStatus.ACTIVE)

    def test_reference_flag_reflects_actual_available_reference(self):
        no_reference = VerifyGoalTool(
            make_session(with_reference=False), FakeVerifier()
        ).call({"claim": "done", "observation_id": "obs_000001"})
        self.assertFalse(no_reference.data["goal_reference_used"])
        disabled = VerifyGoalTool(make_session(), FakeVerifier()).call(
            {
                "claim": "done",
                "observation_id": "obs_000001",
                "use_goal_reference": False,
            }
        )
        self.assertFalse(disabled.data["goal_reference_used"])

    def test_stale_observation_is_rejected_before_verifier(self):
        session = make_session()
        session.add_environment_steps(1)
        verifier = FakeVerifier()
        result = VerifyGoalTool(session, verifier).call(
            {"claim": "done", "observation_id": "obs_000001"}
        )
        self.assertEqual(result.error.code, ErrorCode.STALE_OBSERVATION)
        self.assertEqual(verifier.calls, [])

    def test_verifier_unavailable_is_structured(self):
        result = VerifyGoalTool(
            make_session(),
            FakeVerifier(error=GoalVerifierUnavailable("oracle offline")),
        ).call({"claim": "done", "observation_id": "obs_000001"})
        self.assertEqual(result.status, ToolStatus.UNAVAILABLE)
        self.assertEqual(result.error.code, ErrorCode.VERIFIER_UNAVAILABLE)

    def test_claim_cannot_override_failure(self):
        session = make_session()
        result = VerifyGoalTool(session, FakeVerifier("failure")).call(
            {
                "claim": "I insist that everything succeeded",
                "observation_id": "obs_000001",
            }
        )
        self.assertEqual(result.data["status"], "failure")
        self.assertEqual(session.status, AgentSessionStatus.ACTIVE)


class AbortGoalToolTest(unittest.TestCase):
    def test_abort_records_reason_summary_and_latest_observation(self):
        session = make_session()
        result = AbortGoalTool(session).call(
            {
                "reason_code": "repeated_execution_failure",
                "summary": "three attempts failed and replanning found no alternative",
                "last_observation_id": "obs_000001",
            },
            "abort-once",
        )
        self.assertEqual(result.status, ToolStatus.SUCCEEDED)
        self.assertEqual(result.data["status"], "aborted_by_agent")
        self.assertEqual(session.status, AgentSessionStatus.ABORTED_BY_AGENT)
        self.assertEqual(session.termination_reason, "repeated_execution_failure")
        self.assertEqual(
            session.termination_details["last_observation_id"], "obs_000001"
        )
        replay = AbortGoalTool(session).call({}, "abort-once")
        self.assertIs(replay, result)

    def test_abort_defaults_to_latest_observation(self):
        session = make_session()
        session.add_observation("artifact://new/static", "artifact://new/wrist")
        result = AbortGoalTool(session).call(
            {
                "reason_code": "task_infeasible",
                "summary": "the requested state is physically impossible",
            }
        )
        self.assertEqual(result.data["last_observation_id"], "obs_000002")

    def test_non_latest_observation_is_rejected(self):
        session = make_session()
        session.add_observation("artifact://new/static", "artifact://new/wrist")
        result = AbortGoalTool(session).call(
            {
                "reason_code": "insufficient_observation",
                "summary": "camera evidence is insufficient",
                "last_observation_id": "obs_000001",
            }
        )
        self.assertEqual(result.error.code, ErrorCode.STALE_OBSERVATION)
        self.assertEqual(session.status, AgentSessionStatus.ACTIVE)

    def test_unknown_reason_is_rejected_without_terminating(self):
        session = make_session()
        result = AbortGoalTool(session).call(
            {"reason_code": "give_up", "summary": "no"}
        )
        self.assertEqual(result.error.code, ErrorCode.INVALID_ARGUMENT)
        self.assertEqual(session.status, AgentSessionStatus.ACTIVE)

    def test_new_abort_after_terminal_session_is_rejected(self):
        session = make_session()
        tool = AbortGoalTool(session)
        tool.call(
            {"reason_code": "unsafe_to_continue", "summary": "unsafe"},
            "first",
        )
        second = tool.call(
            {"reason_code": "unsafe_to_continue", "summary": "again"},
            "second",
        )
        self.assertEqual(second.error.code, ErrorCode.SESSION_TERMINAL)

    def test_runtime_can_require_attempt_exhaustion_for_execution_failure(self):
        session = make_session()
        tool = AbortGoalTool(session, enforce_attempt_exhaustion=True)
        rejected = tool.call(
            {
                "reason_code": "repeated_execution_failure",
                "summary": "one attempt is not enough evidence",
            }
        )
        self.assertEqual(rejected.error.code, ErrorCode.INVALID_PRECONDITION)
        self.assertEqual(session.status, AgentSessionStatus.ACTIVE)

        subtask_id = session.resolve_logical_subtask("place_in_drawer", ())
        for _ in range(session.goal.limits.skill_attempt_limit):
            session.consume_attempt(subtask_id)
        accepted = tool.call(
            {
                "reason_code": "repeated_execution_failure",
                "summary": "the complete attempt budget was exhausted",
            }
        )
        self.assertEqual(accepted.status, ToolStatus.SUCCEEDED)
        self.assertEqual(session.status, AgentSessionStatus.ABORTED_BY_AGENT)


class FakeTaskOracle:
    def __init__(self, achieved):
        self.achieved = set(achieved)
        self.calls = []

    def get_task_info_for_set(self, start_info, current_info, tasks):
        self.calls.append((start_info, current_info, tasks))
        return self.achieved & set(tasks)


class CalvinOracleGoalVerifierTest(unittest.TestCase):
    def test_all_predicates_must_be_satisfied(self):
        adapter = type(
            "Adapter", (), {"get_info": lambda self: {"state": "current"}}
        )()
        session = make_session(predicates=["open_drawer", "place_in_drawer"])
        partial = CalvinOracleGoalVerifier(
            adapter,
            FakeTaskOracle({"open_drawer"}),
            start_info={"state": "start"},
        ).verify(session.goal, session.latest_observation)
        self.assertEqual(partial.status, "failure")
        self.assertEqual(
            partial.evidence["predicates"],
            [
                {"name": "open_drawer", "satisfied": True},
                {"name": "place_in_drawer", "satisfied": False},
            ],
        )
        complete = CalvinOracleGoalVerifier(
            adapter,
            FakeTaskOracle({"open_drawer", "place_in_drawer"}),
            start_info={"state": "start"},
        ).verify(session.goal, session.latest_observation)
        self.assertEqual(complete.status, "success")


if __name__ == "__main__":
    unittest.main()
