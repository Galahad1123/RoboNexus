import unittest

import numpy as np

from backends.calvin.execution.skill_executor import CalvinSkillExecutor
from calvin.evaluation.remote_policy.calvin_env_adapter import CalvinEnvAdapter


class _FakeRobot:
    def __init__(self):
        self.gripper_action = 1
        self.restored = None

    def reset(self, robot_obs):
        self.restored = np.asarray(robot_obs).copy()


class _FakeEnv:
    def __init__(self, robot_obs):
        self.robot = _FakeRobot()
        self.robot_obs = np.asarray(robot_obs).copy()
        self.scene_marker = object()
        self.full_reset_called = False

    def reset(self, **kwargs):
        self.full_reset_called = True
        raise AssertionError("retry recovery must not reset the scene")

    def get_obs(self):
        restored = self.robot.restored.copy()
        restored[14] = self.robot.gripper_action
        self.robot_obs = restored
        return {
            "robot_obs": restored,
            "rgb_obs": {
                "rgb_static": np.zeros((4, 4, 3), dtype=np.uint8),
                "rgb_gripper": np.zeros((4, 4, 3), dtype=np.uint8),
            },
        }


def _metadata():
    return {
        "image_keys": [
            "observation.images.static",
            "observation.images.gripper",
        ],
        "image_shapes": {
            "observation.images.static": [3, 4, 4],
            "observation.images.gripper": [3, 4, 4],
        },
        "state_dim": 15,
        "action_dim": 7,
    }


class RetryRobotRestoreTest(unittest.TestCase):
    def test_adapter_restores_robot_without_resetting_scene_or_step_counter(self):
        initial = np.arange(15, dtype=np.float64)
        initial[14] = -1
        env = _FakeEnv(np.zeros(15))
        adapter = CalvinEnvAdapter(env, _metadata())
        adapter.environment_steps = 17
        scene_marker = env.scene_marker

        observation = adapter.restore_robot(initial)

        np.testing.assert_array_equal(env.robot.restored, initial)
        self.assertEqual(env.robot.gripper_action, -1)
        self.assertEqual(observation["robot_obs"][14], -1)
        self.assertIs(env.scene_marker, scene_marker)
        self.assertFalse(env.full_reset_called)
        self.assertEqual(adapter.environment_steps, 17)

    def test_executor_restores_only_before_retry(self):
        initial = np.arange(15, dtype=np.float64)
        initial[14] = 1
        env = _FakeEnv(np.zeros(15))
        adapter = CalvinEnvAdapter(env, _metadata())
        adapter.current_observation = {"robot_obs": np.zeros(15)}
        executor = object.__new__(CalvinSkillExecutor)
        executor.adapter = adapter
        executor.initial_robot_observation = initial

        first = executor._restore_robot_for_retry("open_drawer", 1)
        self.assertFalse(first["performed"])
        self.assertIsNone(env.robot.restored)

        retry = executor._restore_robot_for_retry("open_drawer", 2)
        self.assertTrue(retry["performed"])
        self.assertFalse(retry["scene_reset"])
        self.assertAlmostEqual(retry["max_joint_error_after_rad"], 0.0)
        np.testing.assert_array_equal(env.robot.restored, initial)

    def test_stateful_place_retry_preserves_current_held_state(self):
        initial = np.zeros(15, dtype=np.float64)
        initial[14] = 1
        env = _FakeEnv(initial)
        adapter = CalvinEnvAdapter(env, _metadata())
        adapter.current_observation = {"robot_obs": initial.copy()}
        executor = object.__new__(CalvinSkillExecutor)
        executor.adapter = adapter
        executor.initial_robot_observation = initial

        recovery = executor._restore_robot_for_retry("place_in_drawer", 2)

        self.assertFalse(recovery["performed"])
        self.assertEqual(
            recovery["reason"], "preserve_stateful_skill_robot_state"
        )
        self.assertIsNone(env.robot.restored)


if __name__ == "__main__":
    unittest.main()
