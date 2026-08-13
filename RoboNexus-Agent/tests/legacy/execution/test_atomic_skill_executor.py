"""Tests for the legacy atomic-skill executor."""

import unittest

import numpy as np

from core.legacy.execution import AtomicSkillExecutionError, AtomicSkillExecutor
from core.logging import InMemoryEventLogger
from core.policies import PolicyClientPool, PolicyRouter, load_policy_catalog
from core.skills import load_skill_registry
from tests.paths import AGENT_ROOT


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


class FakePolicyClient:
    def __init__(self, policy):
        self.policy = policy
        self.metadata = metadata()
        self.reset_calls = []
        self.step_calls = []
        self.closed = False
        self.action = np.arange(7, dtype=np.float32)

    def reset(self, episode_id, task, replan_interval, episode_seed):
        self.reset_calls.append(
            {
                "episode_id": episode_id,
                "task": task,
                "replan_interval": replan_interval,
                "episode_seed": episode_seed,
            }
        )
        return {
            "type": "reset_ack",
            "ok": True,
            "episode_seed": episode_seed,
            "replan_interval": replan_interval,
        }

    def step(self, episode_id, observation, step):
        self.step_calls.append((episode_id, observation, step))
        return {
            "type": "action",
            "action": self.action,
            "metadata": {"replanned": not self.step_calls[:-1]},
        }

    def health(self):
        return {"type": "health", "ok": True}

    def close(self):
        self.closed = True


class AtomicSkillExecutorTest(unittest.TestCase):
    def setUp(self):
        registry = load_skill_registry(AGENT_ROOT / "configs/calvin/skills_d34.yaml")
        catalog = load_policy_catalog(
            AGENT_ROOT / "configs/calvin/policies.yaml",
            required_policy_ids=registry.policy_groups(),
        )
        self.created = {}

        def factory(policy):
            client = FakePolicyClient(policy)
            self.created[policy.policy_id] = client
            return client

        self.pool = PolicyClientPool(catalog, client_factory=factory)
        self.router = PolicyRouter(registry, catalog)
        self.events = InMemoryEventLogger()
        self.executor = AtomicSkillExecutor(self.router, self.pool, self.events)

    def tearDown(self):
        self.pool.close_all()

    def test_begin_routes_skill_and_uses_registry_instruction(self):
        context = self.executor.begin_attempt(
            "push_red_block_left", "goal-7", attempt=1, episode_seed=19
        )
        client = self.created["push"]
        self.assertEqual(context.policy_id, "push")
        self.assertEqual(context.policy_episode_id, "goal-7:push_red_block_left:attempt-1")
        self.assertEqual(
            client.reset_calls[0]["task"], "push the red block to the left"
        )
        self.assertEqual(client.reset_calls[0]["replan_interval"], 16)
        event = self.events.events[0]
        self.assertEqual(event["event"], "atomic_attempt_started")
        self.assertTrue(event["scene_continuation"])
        self.assertEqual(event["endpoint"], "ws://127.0.0.1:10094")

    def test_step_returns_owned_action_and_checks_monotonic_steps(self):
        self.executor.begin_attempt("open_drawer", "goal-1", 1, 0)
        observation = {"scene": "current"}
        result = self.executor.step(observation, 12)
        self.assertEqual(result.policy_id, "general")
        np.testing.assert_array_equal(result.action, np.arange(7, dtype=np.float32))
        self.created["general"].action[:] = -1
        np.testing.assert_array_equal(result.action, np.arange(7, dtype=np.float32))
        self.assertIs(self.created["general"].step_calls[0][1], observation)
        with self.assertRaisesRegex(AtomicSkillExecutionError, "step_index must increase"):
            self.executor.step(observation, 12)

    def test_retry_resets_policy_but_accepts_current_scene_observation(self):
        current_scene = {"state": 0}
        self.executor.begin_attempt("open_drawer", "goal-2", 1, 5)
        self.executor.step(current_scene, 0)
        first = self.executor.end_attempt(False, 1, "oracle_failure")
        current_scene["state"] = 1

        second_context = self.executor.begin_attempt("open_drawer", "goal-2", 2, 6)
        self.executor.step(current_scene, 1)
        second = self.executor.end_attempt(True, 1, "oracle_success")

        client = self.created["general"]
        self.assertEqual(len(client.reset_calls), 2)
        self.assertIs(client.step_calls[-1][1], current_scene)
        self.assertEqual(client.step_calls[-1][1]["state"], 1)
        self.assertFalse(first.success)
        self.assertTrue(second.success)
        self.assertEqual(second_context.attempt, 2)

    def test_cannot_overlap_attempts_or_step_while_idle(self):
        with self.assertRaisesRegex(AtomicSkillExecutionError, "No atomic skill"):
            self.executor.step({}, 0)
        self.executor.begin_attempt("open_drawer", "goal-3", 1, 0)
        with self.assertRaisesRegex(AtomicSkillExecutionError, "still active"):
            self.executor.begin_attempt("close_drawer", "goal-3", 1, 0)

    def test_invalid_action_dimension_is_rejected(self):
        self.executor.begin_attempt("stack_block", "goal-4", 1, 0)
        self.created["stack"].action = np.zeros(8, dtype=np.float32)
        with self.assertRaisesRegex(AtomicSkillExecutionError, "action dim 8, expected 7"):
            self.executor.step({}, 0)


if __name__ == "__main__":
    unittest.main()
