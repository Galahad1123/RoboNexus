import sys
from pathlib import Path
import unittest


WORKSPACE = Path(__file__).resolve().parents[2]
REMOTE_POLICY = WORKSPACE / "calvin" / "evaluation" / "remote_policy"
sys.path.insert(0, str(REMOTE_POLICY))

from task_suite import ConfiguredCalvinSuite  # noqa: E402
from rollout_engine import derive_episode_seed, restoration_metrics  # noqa: E402
from calvin_env_adapter import CalvinEnvAdapter  # noqa: E402
from model_client import _pack, _unpack  # noqa: E402


class ConfiguredCalvinSuiteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = ConfiguredCalvinSuite(
            WORKSPACE / "datasets" / "calvin_d_10" / "configs" / "evaluation.yaml"
        )

    def test_loads_all_ten_tasks(self):
        self.assertEqual(len(self.suite.task_names()), 10)
        self.assertIn("stack_block", self.suite.task_names())

    def test_episode_selection_is_deterministic(self):
        first = self.suite.episodes("place_in_drawer", count=3, seed=42)
        second = self.suite.episodes("place_in_drawer", count=3, seed=42)
        self.assertEqual(
            [episode.episode_id for episode in first],
            [episode.episode_id for episode in second],
        )
        self.assertTrue(all(episode.source_frame.is_file() for episode in first))

    def test_instructions_and_oracles_come_from_suite_config(self):
        episode = self.suite.episodes("lift_blue_block_drawer", count=1)[0]
        self.assertEqual(episode.instruction, "lift the blue block from the drawer")
        self.assertEqual(episode.oracle_task, "lift_blue_block_drawer")
        robot_obs, scene_obs = episode.load_initial_state()
        self.assertEqual(robot_obs.shape, (15,))
        self.assertEqual(scene_obs.shape, (24,))

    def test_restore_metrics_ignore_command_and_handle_angle_wrap(self):
        import numpy as np

        robot = np.zeros(15)
        scene = np.zeros(24)
        restored_robot = robot.copy()
        restored_scene = scene.copy()
        restored_robot[14] = -1
        robot[14] = 1
        restored_scene[9] = 2 * np.pi
        metrics = restoration_metrics(
            robot, scene, restored_robot, restored_scene
        )
        self.assertEqual(metrics["gripper_command_expected"], 1)
        self.assertEqual(metrics["gripper_command_restored"], -1)
        self.assertAlmostEqual(metrics["object_orientation_rad"], 0.0, places=6)

    def test_transport_action_is_safe_for_calvin_in_place_scaling(self):
        import numpy as np

        class MutatingEnv:
            def step(self, action):
                action[:3] *= 0.02
                return action

        source = np.asarray([2.0, -2.0, 0, 0, 0, 0, 0.25], dtype=np.float32)
        decoded = _unpack(_pack({"action": source}))["action"]
        adapter = CalvinEnvAdapter.__new__(CalvinEnvAdapter)
        adapter.env = MutatingEnv()
        result = adapter.step(decoded)
        self.assertTrue(result.flags.writeable)
        self.assertTrue(np.allclose(result[:3], [0.02, -0.02, 0.0]))
        self.assertEqual(adapter.last_executed_action[0], 1.0)
        self.assertEqual(adapter.last_executed_action[1], -1.0)
        self.assertEqual(adapter.last_executed_action[6], 1.0)

    def test_gripper_action_is_binarized(self):
        import numpy as np

        close = CalvinEnvAdapter.prepare_action(
            np.asarray([0, 0, 0, 0, 0, 0, -0.001], dtype=np.float32)
        )
        opened = CalvinEnvAdapter.prepare_action(
            np.asarray([0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32)
        )
        self.assertEqual(close[6], -1.0)
        self.assertEqual(opened[6], 1.0)

    def test_episode_seed_is_stable_and_episode_specific(self):
        first = derive_episode_seed(42, "validation-000117")
        self.assertEqual(first, derive_episode_seed(42, "validation-000117"))
        self.assertNotEqual(first, derive_episode_seed(42, "validation-000118"))
        self.assertNotEqual(first, derive_episode_seed(43, "validation-000117"))


if __name__ == "__main__":
    unittest.main()
