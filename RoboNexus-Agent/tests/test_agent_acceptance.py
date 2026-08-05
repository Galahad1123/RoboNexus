import unittest
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from backends.calvin.execution.goal_attempt_runner import CalvinGoalAttemptRunner
from backends.calvin.verification.state_inspector import CalvinStateInspector
from core.logging import InMemoryEventLogger
from core.legacy.orchestration import GoalExecutionConfig, GoalOrchestrator, GoalStatus
from core.legacy.planning import GoalSpec, ScriptedPlanner
from core.policies import PolicyClientPool, PolicyRouter, load_policy_catalog
from core.skills import load_skill_registry


WORKSPACE = Path(__file__).resolve().parents[2]
AGENT_ROOT = WORKSPACE / "RoboNexus-Agent"


def metadata():
    return {
        "protocol_version": 2,
        "policy": "multitask_dit_policy",
        "image_keys": [
            "observation.images.static",
            "observation.images.gripper",
        ],
        "image_shapes": {
            "observation.images.static": [3, 200, 200],
            "observation.images.gripper": [3, 200, 200],
        },
        "state_key": "observation.state",
        "state_dim": 15,
        "action_dim": 7,
        "n_obs_steps": 2,
        "model_action_chunk_size": 16,
        "horizon": 32,
    }


class AcceptanceAdapter:
    def __init__(self, observation):
        self.observation = observation
        self.last_executed_action = np.zeros(7, dtype=np.float32)
        self.info = {"succeeded": False}

    def get_info(self):
        return dict(self.info)

    @staticmethod
    def observation_for_policy(observation, **kwargs):
        return {"observation.state": np.zeros(15, dtype=np.float32)}

    def step(self, action):
        self.last_executed_action = np.asarray(action, dtype=np.float32).copy()
        self.info = {"succeeded": bool(self.last_executed_action[0] > 0.5)}
        return self.observation, 0, False, self.get_info()


class AcceptanceOracle:
    @staticmethod
    def get_task_info_for_set(start, current, tasks):
        return set(tasks) if current["succeeded"] else set()


class AcceptancePolicyClient:
    def __init__(self, policy, instruction_to_skill, outcomes, reset_log):
        self.policy = policy
        self.metadata = metadata()
        self.instruction_to_skill = instruction_to_skill
        self.outcomes = outcomes
        self.reset_log = reset_log
        self.current_success = False
        self.closed = False

    def reset(self, episode_id, task, replan_interval, episode_seed):
        skill_id = self.instruction_to_skill[task]
        self.current_success = self.outcomes[skill_id].popleft()
        self.reset_log.append((self.policy.policy_id, skill_id, episode_id))
        return {
            "type": "reset_ack",
            "ok": True,
            "episode_seed": episode_seed,
            "replan_interval": replan_interval,
        }

    def step(self, episode_id, observation, step):
        action = np.zeros(7, dtype=np.float32)
        action[0] = 1.0 if self.current_success else 0.0
        return {
            "type": "action",
            "action": action,
            "metadata": {"replanned": True},
        }

    def health(self):
        return {"type": "health", "ok": True}

    def close(self):
        self.closed = True


class RecordingMaskProvider:
    def __init__(self):
        self.reset_episodes = []
        self.mask_calls = 0

    def reset(self, episode):
        self.reset_episodes.append(episode)

    def masks(self, observation):
        self.mask_calls += 1
        return {}


class AgentAcceptanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_skill_registry(
            AGENT_ROOT / "configs/calvin/skills_d34.yaml"
        )
        cls.catalog = load_policy_catalog(
            AGENT_ROOT / "configs/calvin/policies.yaml",
            required_policy_ids=cls.registry.policy_groups(),
        )
        cls.instruction_to_skill = {
            cls.registry.get(skill_id).instruction: skill_id
            for skill_id in cls.registry.skill_ids()
        }

    def run_acceptance(
        self,
        steps,
        outcomes,
        replan_scripts=(),
        scene_obs=None,
        inspect=False,
        mask_provider=None,
        mask_roles_by_skill=None,
    ):
        queues = defaultdict(deque)
        for skill_id, values in outcomes.items():
            queues[skill_id].extend(values)
        reset_log = []

        def factory(policy):
            return AcceptancePolicyClient(
                policy,
                self.instruction_to_skill,
                queues,
                reset_log,
            )

        pool = PolicyClientPool(self.catalog, client_factory=factory)
        router = PolicyRouter(self.registry, self.catalog)
        planner = ScriptedPlanner(
            steps, replan_scripts, skill_registry=self.registry
        )
        scene = (
            np.asarray(scene_obs, dtype=np.float32).copy()
            if scene_obs is not None
            else np.zeros(24, dtype=np.float32)
        )
        observation = {
            "scene_obs": scene,
            "robot_obs": np.zeros(15, dtype=np.float32),
        }
        adapter = AcceptanceAdapter(observation)
        events = InMemoryEventLogger()
        attempts = CalvinGoalAttemptRunner(
            adapter=adapter,
            task_oracle=AcceptanceOracle(),
            skill_registry=self.registry,
            policy_router=router,
            client_pool=pool,
            event_logger=events,
            mask_provider=mask_provider,
            mask_roles_by_skill=mask_roles_by_skill,
            max_steps_override=1,
        )
        orchestrator = GoalOrchestrator(
            planner,
            attempts,
            GoalExecutionConfig(),
            events,
            state_inspector=CalvinStateInspector() if inspect else None,
        )
        try:
            result = orchestrator.run(
                GoalSpec("acceptance-goal", "scripted acceptance"), observation
            )
        finally:
            pool.close_all()
        return result, planner, reset_log, events

    def test_01_single_skill_single_policy(self):
        result, _, resets, _ = self.run_acceptance(
            ["open_drawer"], {"open_drawer": [True]}
        )
        self.assertTrue(result.success)
        self.assertEqual([(item[0], item[1]) for item in resets], [
            ("general", "open_drawer")
        ])

    def test_02_two_skills_same_policy(self):
        result, _, resets, _ = self.run_acceptance(
            ["open_drawer", "close_drawer"],
            {"open_drawer": [True], "close_drawer": [True]},
        )
        self.assertTrue(result.success)
        self.assertEqual([item[0] for item in resets], ["general", "general"])

    def test_03_two_skills_cross_policy(self):
        result, _, resets, _ = self.run_acceptance(
            ["open_drawer", "push_red_block_left"],
            {"open_drawer": [True], "push_red_block_left": [True]},
        )
        self.assertTrue(result.success)
        self.assertEqual([item[0] for item in resets], ["general", "push"])

    def test_04_attempt_failure_then_success(self):
        result, _, resets, _ = self.run_acceptance(
            ["open_drawer"], {"open_drawer": [False, True]}
        )
        self.assertTrue(result.success)
        self.assertEqual(result.attempts_executed, 2)
        self.assertEqual(len(resets), 2)

    def test_05_three_failures_trigger_replan(self):
        result, _, resets, _ = self.run_acceptance(
            ["open_drawer"],
            {
                "open_drawer": [False, False, False],
                "close_drawer": [True],
            },
            replan_scripts=[["close_drawer"]],
        )
        self.assertTrue(result.success)
        self.assertEqual(result.replans_used, 1)
        self.assertEqual(len(resets), 4)

    def test_06_two_replans_then_abort(self):
        result, _, resets, _ = self.run_acceptance(
            ["open_drawer"],
            {"open_drawer": [False] * 9},
            replan_scripts=[["open_drawer"], ["open_drawer"]],
        )
        self.assertEqual(result.status, GoalStatus.ABORTED)
        self.assertEqual(result.termination_reason, "replan_budget_exhausted")
        self.assertEqual(result.replans_used, 2)
        self.assertEqual(len(resets), 9)

    def test_07_already_satisfied_skill_is_skipped(self):
        scene = np.zeros(24, dtype=np.float32)
        scene[1] = 0.22
        result, _, resets, events = self.run_acceptance(
            ["open_drawer"], {}, scene_obs=scene, inspect=True
        )
        self.assertTrue(result.success)
        self.assertEqual(result.attempts_executed, 0)
        self.assertEqual(resets, [])
        self.assertTrue(result.skill_records[0].skipped)
        self.assertIn("skill_skipped", [event["event"] for event in events.events])

    def test_08_completed_goal_does_not_request_planner_again(self):
        result, planner, _, events = self.run_acceptance([], {})
        self.assertTrue(result.success)
        self.assertEqual(planner.request_count, 1)
        self.assertEqual(events.events[-1]["event"], "goal_succeeded")

    def test_09_mask_provider_is_initialized_with_current_skill_roles(self):
        provider = RecordingMaskProvider()
        roles = {"target": [{"fixed": "table", "links": ["drawer_link"]}]}
        result, _, _, _ = self.run_acceptance(
            ["open_drawer"],
            {"open_drawer": [True]},
            mask_provider=provider,
            mask_roles_by_skill={"open_drawer": roles},
        )
        self.assertTrue(result.success)
        self.assertEqual(provider.mask_calls, 1)
        self.assertEqual(len(provider.reset_episodes), 1)
        self.assertEqual(provider.reset_episodes[0].mask_roles, roles)

    def test_10_enabled_mask_provider_rejects_missing_roles(self):
        provider = RecordingMaskProvider()
        with self.assertRaisesRegex(RuntimeError, "has no mask_roles"):
            self.run_acceptance(
                ["open_drawer"],
                {"open_drawer": [True]},
                mask_provider=provider,
            )


if __name__ == "__main__":
    unittest.main()
