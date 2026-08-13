"""Tests for the legacy scripted GoalOrchestrator."""

import unittest
from collections import deque

from core.legacy.execution.schemas import AttemptResult
from core.legacy.inspection import InspectionStatus, SkillInspection
from core.logging import InMemoryEventLogger
from core.legacy.orchestration import GoalExecutionConfig, GoalOrchestrator, GoalStatus
from core.legacy.planning import (
    GoalSpec,
    PlannerDecision,
    PlannerDecisionType,
    ScriptedPlanner,
)
from core.skills import load_skill_registry
from tests.paths import AGENT_ROOT


class FakeAttemptRunner:
    def __init__(self, outcomes):
        self.outcomes = {
            skill_id: deque(values) for skill_id, values in outcomes.items()
        }
        self.requests = []
        self.results = []

    def run_attempt(self, request):
        self.requests.append(request)
        success = self.outcomes[request.skill_id].popleft()
        observation = {
            "sequence": len(self.requests),
            "previous": request.observation,
        }
        result = AttemptResult(
            success=success,
            steps=1,
            termination_reason="oracle_success" if success else "oracle_failure",
            observation=observation,
        )
        self.results.append(result)
        return result


class GoalOrchestratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_skill_registry(
            AGENT_ROOT / "configs/calvin/skills_d34.yaml"
        )

    def run_goal(self, steps, replans, outcomes, config=None, inspector=None):
        planner = ScriptedPlanner(steps, replans, self.registry)
        runner = FakeAttemptRunner(outcomes)
        events = InMemoryEventLogger()
        orchestrator = GoalOrchestrator(
            planner, runner, config, events, state_inspector=inspector
        )
        initial = {"sequence": 0}
        result = orchestrator.run(
            GoalSpec("goal-test", "execute the scripted task"), initial
        )
        return result, planner, runner, events, initial

    def test_two_successful_skills_are_requested_one_at_a_time(self):
        result, planner, runner, _, _ = self.run_goal(
            ["open_drawer", "close_drawer"],
            [],
            {"open_drawer": [True], "close_drawer": [True]},
        )
        self.assertTrue(result.success)
        self.assertEqual(
            result.completed_skill_ids, ("open_drawer", "close_drawer")
        )
        self.assertEqual([item.skill_id for item in runner.requests], [
            "open_drawer",
            "close_drawer",
        ])
        # Two atomic decisions plus one terminal COMPLETE decision.
        self.assertEqual(planner.request_count, 3)

    def test_attempts_continue_from_previous_observation_and_third_succeeds(self):
        result, _, runner, _, initial = self.run_goal(
            ["open_drawer"], [], {"open_drawer": [False, False, True]}
        )
        self.assertTrue(result.success)
        self.assertEqual(result.attempts_executed, 3)
        self.assertIs(runner.requests[0].observation, initial)
        self.assertIs(runner.requests[1].observation, runner.results[0].observation)
        self.assertIs(runner.requests[2].observation, runner.results[1].observation)
        self.assertEqual([request.attempt for request in runner.requests], [1, 2, 3])
        self.assertEqual(len({request.episode_seed for request in runner.requests}), 3)

    def test_three_failures_trigger_replan_and_alternative_skill(self):
        result, _, runner, events, _ = self.run_goal(
            ["open_drawer"],
            [["close_drawer"]],
            {"open_drawer": [False, False, False], "close_drawer": [True]},
        )
        self.assertTrue(result.success)
        self.assertEqual(result.replans_used, 1)
        self.assertEqual(result.completed_skill_ids, ("close_drawer",))
        self.assertEqual(
            [request.skill_id for request in runner.requests],
            ["open_drawer"] * 3 + ["close_drawer"],
        )
        self.assertIn("replan_started", [event["event"] for event in events.events])

    def test_two_replans_then_abort_after_nine_failures(self):
        result, planner, runner, events, _ = self.run_goal(
            ["open_drawer"],
            [["open_drawer"], ["open_drawer"]],
            {"open_drawer": [False] * 9},
        )
        self.assertFalse(result.success)
        self.assertEqual(result.status, GoalStatus.ABORTED)
        self.assertEqual(result.termination_reason, "replan_budget_exhausted")
        self.assertEqual(result.replans_used, 2)
        self.assertEqual(result.attempts_executed, 9)
        self.assertEqual(len(runner.requests), 9)
        self.assertEqual(planner.request_count, 3)
        self.assertEqual(events.events[-1]["event"], "goal_aborted")

    def test_planner_can_abort_when_no_replan_script_exists(self):
        result, _, _, _, _ = self.run_goal(
            ["open_drawer"], [], {"open_drawer": [False] * 3}
        )
        self.assertEqual(result.status, GoalStatus.ABORTED)
        self.assertEqual(result.termination_reason, "scripted_replan_unavailable")
        self.assertEqual(result.replans_used, 1)

    def test_goal_completion_is_terminal_and_does_not_request_planner_again(self):
        result, planner, _, events, _ = self.run_goal([], [], {})
        self.assertTrue(result.success)
        self.assertEqual(planner.request_count, 1)
        self.assertEqual(events.events[-1]["event"], "goal_succeeded")

    def test_planner_decision_enforces_one_atomic_skill(self):
        with self.assertRaisesRegex(ValueError, "one non-empty skill_id"):
            PlannerDecision(PlannerDecisionType.EXECUTE, skill_id=None)
        with self.assertRaisesRegex(ValueError, "cannot contain skill_id"):
            PlannerDecision(PlannerDecisionType.COMPLETE, skill_id="open_drawer")

    def test_default_budget_is_three_attempts_and_two_replans(self):
        config = GoalExecutionConfig()
        self.assertEqual(config.max_attempts_per_skill, 3)
        self.assertEqual(config.max_replans, 2)

    def test_satisfied_skill_is_skipped_without_running_attempt(self):
        class Inspector:
            @staticmethod
            def inspect(skill_id, observation):
                return SkillInspection(
                    InspectionStatus.SATISFIED, "already_satisfied"
                )

        result, planner, runner, events, _ = self.run_goal(
            ["open_drawer"], [], {}, inspector=Inspector()
        )
        self.assertTrue(result.success)
        self.assertEqual(result.completed_skill_ids, ("open_drawer",))
        self.assertEqual(result.attempts_executed, 0)
        self.assertEqual(runner.requests, [])
        self.assertTrue(result.skill_records[0].skipped)
        self.assertEqual(planner.request_count, 2)
        self.assertIn("skill_skipped", [event["event"] for event in events.events])

    def test_invalid_precondition_triggers_replan_without_attempt(self):
        class Inspector:
            @staticmethod
            def inspect(skill_id, observation):
                if skill_id == "place_in_drawer":
                    return SkillInspection(
                        InspectionStatus.INVALID_PRECONDITION,
                        "drawer_must_be_open",
                    )
                return SkillInspection(InspectionStatus.READY, "ready")

        result, _, runner, events, _ = self.run_goal(
            ["place_in_drawer"],
            [["open_drawer"]],
            {"open_drawer": [True]},
            inspector=Inspector(),
        )
        self.assertTrue(result.success)
        self.assertEqual(result.replans_used, 1)
        self.assertEqual(result.attempts_executed, 1)
        self.assertEqual([request.skill_id for request in runner.requests], ["open_drawer"])
        invalid = [
            event
            for event in events.events
            if event["event"] == "skill_precondition_invalid"
        ]
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0]["reason"], "drawer_must_be_open")

    def test_invalid_preconditions_obey_replan_budget(self):
        class Inspector:
            @staticmethod
            def inspect(skill_id, observation):
                return SkillInspection(
                    InspectionStatus.INVALID_PRECONDITION,
                    "drawer_must_be_open",
                )

        result, _, runner, _, _ = self.run_goal(
            ["place_in_drawer"],
            [["place_in_drawer"], ["place_in_drawer"]],
            {},
            inspector=Inspector(),
        )
        self.assertEqual(result.status, GoalStatus.ABORTED)
        self.assertEqual(result.termination_reason, "replan_budget_exhausted")
        self.assertEqual(result.replans_used, 2)
        self.assertEqual(result.attempts_executed, 0)
        self.assertEqual(runner.requests, [])


if __name__ == "__main__":
    unittest.main()
