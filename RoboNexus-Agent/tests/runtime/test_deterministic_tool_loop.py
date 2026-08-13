"""Tests for the retained deterministic tool-loop harness."""

import unittest

from core.domain import (
    AgentSessionStatus,
    GoalInput,
    ObservationRef,
    ToolStatus,
)
from core.runtime import AgentSession, DeterministicLoopConfig, DeterministicToolLoop


def make_session():
    return AgentSession(
        GoalInput(
            "goal-loop",
            "lift the red block",
            {"predicates": [{"name": "lift_red_block_table"}]},
            ObservationRef(
                "obs_000001", 0, 0, "artifact://initial/static.png", "artifact://initial/wrist.png"
            ),
        )
    )


class FakeTool:
    def __init__(self, session, statuses, data=None, on_call=None):
        self.session = session
        self.statuses = list(statuses)
        self.data = data or {}
        self.on_call = on_call
        self.calls = []

    def call(self, params):
        self.calls.append(params)
        status = self.statuses.pop(0) if self.statuses else ToolStatus.SUCCEEDED
        if self.on_call is not None:
            self.on_call(len(self.calls), status)
        return self.session.result(
            "fake-{}".format(sum(len(item.calls) for item in ALL_TOOLS)),
            status,
            data=self.data(len(self.calls)) if callable(self.data) else dict(self.data),
        )


ALL_TOOLS = []


class DeterministicToolLoopTest(unittest.TestCase):
    def setUp(self):
        ALL_TOOLS.clear()

    def make_loop(self, execute_statuses, verification_statuses):
        session = make_session()
        inspect = FakeTool(session, [ToolStatus.SUCCEEDED] * 8)
        ground = FakeTool(
            session,
            [ToolStatus.SUCCEEDED] * 8,
            {"groundings": [{"grounding_id": "ground-1", "role": "source"}]},
        )
        execute = FakeTool(session, execute_statuses)
        approach = FakeTool(session, [ToolStatus.SUCCEEDED] * 2)

        def verify_data(call_index):
            status = verification_statuses[call_index - 1]
            if status == "success":
                session.mark_succeeded("success", "verified")
            return {"status": status}

        verify = FakeTool(
            session, [ToolStatus.SUCCEEDED] * len(verification_statuses), verify_data
        )

        def abort_session(call_index, status):
            del call_index, status
            session.abort(AgentSessionStatus.ABORTED_BY_AGENT, "test_abort")

        abort = FakeTool(session, [ToolStatus.SUCCEEDED], on_call=abort_session)
        ALL_TOOLS.extend((inspect, ground, execute, approach, verify, abort))
        return (
            session,
            DeterministicToolLoop(
                session, inspect, ground, execute, approach, verify, abort
            ),
            (inspect, ground, execute, approach, verify, abort),
        )

    @staticmethod
    def config(max_attempts=3):
        return DeterministicLoopConfig(
            "lift_red_block_table",
            (
                {
                    "role": "source",
                    "representation": "semantic",
                    "semantic_id": "block_red",
                },
            ),
            max_attempts=max_attempts,
        )

    def test_first_attempt_success_stops_after_verify(self):
        session, loop, tools = self.make_loop(
            [ToolStatus.SUCCEEDED], ["success"]
        )
        result = loop.run(self.config())
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["successful_attempt"], 1)
        self.assertEqual([item["tool"] for item in result["trace"]], [
            "inspect_scene", "ground_targets", "execute_skill", "verify_goal"
        ])
        self.assertEqual(session.status, AgentSessionStatus.SUCCEEDED)
        self.assertEqual(len(tools[3].calls), 0)
        self.assertEqual(len(tools[5].calls), 0)

    def test_failed_attempt_recovers_and_regrounds_before_retry(self):
        session, loop, tools = self.make_loop(
            [ToolStatus.FAILED, ToolStatus.SUCCEEDED], ["failure", "success"]
        )
        result = loop.run(self.config())
        self.assertEqual(result["successful_attempt"], 2)
        self.assertEqual([item["tool"] for item in result["trace"]], [
            "inspect_scene",
            "ground_targets",
            "execute_skill",
            "verify_goal",
            "inspect_scene",
            "ground_targets",
            "approach_target",
            "inspect_scene",
            "ground_targets",
            "execute_skill",
            "verify_goal",
        ])
        self.assertEqual(len(tools[2].calls), 2)
        self.assertEqual(len(tools[3].calls), 1)
        self.assertEqual(session.status, AgentSessionStatus.SUCCEEDED)

    def test_three_failures_abort_goal(self):
        session, loop, tools = self.make_loop(
            [ToolStatus.FAILED] * 3, ["failure"] * 3
        )
        result = loop.run(self.config())
        self.assertEqual(result["status"], "aborted_by_agent")
        self.assertEqual(len(tools[2].calls), 3)
        self.assertEqual(len(tools[3].calls), 2)
        self.assertEqual(len(tools[5].calls), 1)
        self.assertEqual(session.status, AgentSessionStatus.ABORTED_BY_AGENT)


if __name__ == "__main__":
    unittest.main()
