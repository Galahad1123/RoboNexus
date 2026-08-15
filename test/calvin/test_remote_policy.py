import sys
from pathlib import Path
import unittest

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
REMOTE_POLICY = WORKSPACE / "calvin" / "evaluation" / "remote_policy"
sys.path.insert(0, str(REMOTE_POLICY))
sys.path.insert(0, str(WORKSPACE))

from task_suite import ConfiguredCalvinSuite  # noqa: E402
from rollout_engine import (  # noqa: E402
    derive_episode_seed,
    restoration_metrics,
    success_guard_satisfied,
)
from calvin_env_adapter import CalvinEnvAdapter  # noqa: E402
from model_client import _pack, _unpack  # noqa: E402
from masking import CalvinGTMaskProvider, MaskOverlay, MaskPerturbation  # noqa: E402
from backends.calvin.execution.retry_runner import CalvinOracleRetryAgent  # noqa: E402
from backends.calvin.execution.attempt_runner import CalvinSkillAttemptRunner  # noqa: E402
from backends.calvin.verification.task_verifier import CalvinOracleTaskVerifier  # noqa: E402
from backends.calvin.recovery.motion_controller import (  # noqa: E402
    CalvinCartesianApproachController,
)


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
        self.assertTrue(episode.retry["enabled"])
        self.assertEqual(episode.retry["target"], {"movable": "block_blue"})
        robot_obs, scene_obs = episode.load_initial_state()
        self.assertEqual(robot_obs.shape, (15,))
        self.assertEqual(scene_obs.shape, (24,))
        self.assertEqual(
            episode.mask_roles["source"], [{"movable": "block_blue"}]
        )

    def test_multi_skill_recovery_tasks_disable_retry(self):
        episode = self.suite.episodes("stack_block", count=1)[0]
        self.assertFalse(episode.retry["enabled"])
        self.assertEqual(
            episode.retry["reason"], "requires_multi_skill_recovery"
        )

    def test_cartesian_approach_preserves_gripper_and_reaches_target(self):
        class Robot:
            max_rel_pos = 0.02

        class Env:
            robot = Robot()

        class Adapter:
            env = Env()

            def step(self, action):
                self.commands.append(np.asarray(action).copy())
                self.observation["robot_obs"][:3] += action[:3] * 0.02
                self.last_executed_action = np.asarray(action).copy()
                return self.observation, 0, False, {}

        adapter = Adapter()
        adapter.commands = []
        adapter.observation = {
            "robot_obs": np.asarray(
                [0.30, 0.00, 0.55] + [0.0] * 11 + [-1.0],
                dtype=np.float64,
            )
        }
        result = CalvinCartesianApproachController(adapter).execute(
            adapter.observation,
            "block_red",
            np.asarray([0.40, 0.10, 0.50]),
            {
                "xy_threshold_m": 0.05,
                "hover_height_m": 0.10,
                "z_tolerance_m": 0.01,
                "max_motion_steps": 40,
            },
        )
        self.assertTrue(result.success)
        self.assertLessEqual(result.final_xy_distance_m, 0.05)
        self.assertTrue(adapter.commands)
        self.assertTrue(all(command[6] == -1.0 for command in adapter.commands))

    def test_cartesian_approach_does_nothing_when_xy_is_already_close(self):
        class Robot:
            max_rel_pos = 0.02

        class Env:
            robot = Robot()

        class Adapter:
            env = Env()

            @staticmethod
            def step(action):
                raise AssertionError("controller must not move inside XY threshold")

        observation = {
            "robot_obs": np.asarray(
                [0.30, 0.00, 0.55] + [0.0] * 11 + [1.0],
                dtype=np.float64,
            )
        }
        result = CalvinCartesianApproachController(Adapter()).execute(
            observation,
            "block_red",
            np.asarray([0.35, 0.02, 0.50]),
            {"xy_threshold_m": 0.10},
        )
        self.assertTrue(result.success)
        self.assertEqual(result.steps, 0)
        self.assertEqual(result.reason, "already_within_xy_threshold")

    def test_cartesian_approach_restores_fixed_orientation_and_opens_gripper(self):
        class Robot:
            max_rel_pos = 0.02
            max_rel_orn = 0.05

        class Env:
            robot = Robot()

        class Adapter:
            env = Env()

            def __init__(self):
                self.observation = {
                    "robot_obs": np.asarray(
                        [0.30, 0.00, 0.55, 1.50, 0.40, -0.20]
                        + [0.0] * 8
                        + [-1.0],
                        dtype=np.float64,
                    )
                }
                self.commands = []

            def step(self, action):
                action = np.asarray(action)
                self.commands.append(action.copy())
                self.observation["robot_obs"][:3] += action[:3] * 0.02
                self.observation["robot_obs"][3:6] += action[3:6] * 0.05
                self.last_executed_action = action.copy()
                return self.observation, 0, False, {}

        adapter = Adapter()
        target_orientation = np.asarray([3.06, 0.0, 1.46])
        result = CalvinCartesianApproachController(adapter).execute(
            adapter.observation,
            "block_red",
            np.asarray([0.34, 0.01, 0.46]),
            {
                "xy_threshold_m": 0.08,
                "target_orientation_euler": target_orientation,
                "orientation_tolerance_rad": 0.08,
                "gripper_command": 1.0,
                "max_motion_steps": 80,
            },
        )
        self.assertTrue(result.success)
        self.assertLessEqual(result.final_orientation_error_rad, 0.08)
        self.assertTrue(adapter.commands)
        self.assertTrue(all(command[6] == 1.0 for command in adapter.commands))

    def test_mask_overlay_does_not_mutate_clean_image(self):
        import numpy as np

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        overlay = MaskOverlay(alpha=1.0, seed=7)
        augmented = overlay.apply(image, {"source": mask})
        self.assertFalse(np.shares_memory(image, augmented))
        self.assertTrue(np.all(image == 0))
        self.assertTrue(np.all(augmented[mask] == [255, 40, 40]))

    def test_mask_perturbation_is_seed_deterministic(self):
        import numpy as np

        mask = np.ones((12, 12), dtype=bool)
        perturbation = MaskPerturbation(
            max_shift_px=2, morphology_px=-1, dropout_probability=0.2
        )
        first = perturbation.apply(mask, np.random.RandomState(123))
        second = perturbation.apply(mask, np.random.RandomState(123))
        self.assertTrue(np.array_equal(first, second))

    def test_mask_outline_is_opaque_while_fill_remains_translucent(self):
        import numpy as np

        image = np.zeros((12, 12, 3), dtype=np.uint8)
        mask = np.zeros((12, 12), dtype=bool)
        mask[3:9, 3:9] = True
        augmented = MaskOverlay(
            alpha=0.5, outline_width_px=1, seed=7
        ).apply(image, {"source": mask})
        self.assertTrue(np.array_equal(augmented[5, 5], [127, 20, 20]))
        self.assertTrue(np.array_equal(augmented[3, 5], [0, 0, 0]))
        # The outline straddles the segmentation boundary for visibility.
        self.assertTrue(np.array_equal(augmented[2, 5], [0, 0, 0]))

    def test_role_outline_color_and_outer_halo(self):
        import numpy as np

        image = np.zeros((14, 14, 3), dtype=np.uint8)
        mask = np.zeros((14, 14), dtype=bool)
        mask[4:10, 4:10] = True
        augmented = MaskOverlay(
            alpha=0,
            outline_width_px=2,
            outline_colors={"destination": (0, 255, 255)},
            outline_halo_width_px=1,
            outline_halo_color=(255, 255, 255),
        ).apply(image, {"destination": mask})
        self.assertTrue(np.array_equal(augmented[4, 6], [0, 255, 255]))
        self.assertTrue(np.array_equal(augmented[1, 6], [255, 255, 255]))
        self.assertTrue(np.array_equal(augmented[0, 6], [0, 0, 0]))

    def test_inside_outline_does_not_expand_object_mask(self):
        import numpy as np

        image = np.full((12, 12, 3), 17, dtype=np.uint8)
        mask = np.zeros((12, 12), dtype=bool)
        mask[3:9, 3:9] = True
        augmented = MaskOverlay(
            alpha=0,
            outline_width_px=2,
            outline_colors={"source": (160, 255, 0)},
            outline_positions={"source": "inside"},
        ).apply(image, {"source": mask})
        changed = np.any(augmented != image, axis=-1)
        self.assertTrue(changed.any())
        self.assertFalse(changed[~mask].any())
        self.assertTrue(np.array_equal(augmented[3, 5], [160, 255, 0]))
        self.assertTrue(np.array_equal(augmented[5, 5], [17, 17, 17]))

    def test_role_specific_halo_and_dynamic_held_visibility(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        source = np.zeros((16, 16), dtype=bool)
        destination = np.zeros((16, 16), dtype=bool)
        source[5:11, 2:6] = True
        destination[5:11, 10:14] = True
        augmented = MaskOverlay(
            alpha=0,
            outline_width_px=2,
            outline_colors={"source": (0, 0, 0), "destination": (0, 255, 255)},
            outline_halo_width_px=0,
            outline_halo_widths={"destination": 1},
        ).apply(image, {"source": source, "destination": destination})
        self.assertTrue(np.array_equal(augmented[2, 3], [0, 0, 0]))
        self.assertTrue(np.array_equal(augmented[2, 12], [255, 255, 255]))

        segmentation = np.asarray([[7, 8]], dtype=np.int64)
        identities = {(7, None, "until_held"), (8, None, "held")}
        visible = CalvinGTMaskProvider._select(segmentation, identities, {8}, {8})
        self.assertTrue(np.array_equal(visible, [[True, True]]))
        hidden = CalvinGTMaskProvider._select(segmentation, identities, set(), {7, 8})
        self.assertTrue(np.array_equal(hidden, [[False, False]]))

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
        self.assertEqual(adapter.environment_steps, 1)
        self.assertEqual(adapter.current_observation, result[0])

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

    def test_release_success_guard_requires_open_command_and_width(self):
        observation = {"robot_obs": np.asarray([0, 0, 0, 0, 0, 0, 0.06])}
        action_open = np.asarray([0, 0, 0, 0, 0, 0, 1])
        action_close = np.asarray([0, 0, 0, 0, 0, 0, -1])
        guard = {"gripper": "open", "min_width_m": 0.04}
        self.assertTrue(success_guard_satisfied(guard, observation, action_open))
        self.assertFalse(success_guard_satisfied(guard, observation, action_close))
        observation["robot_obs"][6] = 0.02
        self.assertFalse(success_guard_satisfied(guard, observation, action_open))

    def test_episode_seed_is_stable_and_episode_specific(self):
        first = derive_episode_seed(42, "validation-000117")
        self.assertEqual(first, derive_episode_seed(42, "validation-000117"))
        self.assertNotEqual(first, derive_episode_seed(42, "validation-000118"))
        self.assertNotEqual(first, derive_episode_seed(43, "validation-000117"))

    def test_oracle_retry_continues_without_environment_reset(self):
        import numpy as np
        from types import SimpleNamespace

        class Episode:
            episode_id = "validation-test"
            task = "open_drawer"
            instruction = "open the drawer"
            oracle_task = "open_drawer"
            source_frame = Path("/tmp/source.npz")
            settle_steps = 0
            max_steps = 1

            @staticmethod
            def load_initial_state():
                return np.zeros(15), np.zeros(24)

        class Adapter:
            def __init__(self):
                self.reset_count = 0
                self.steps = 0

            def reset(self, robot_obs, scene_obs):
                self.reset_count += 1
                return {
                    "robot_obs": robot_obs.copy(),
                    "scene_obs": scene_obs.copy(),
                    "rgb_obs": {"rgb_static": np.zeros((2, 2, 3), np.uint8)},
                }

            def get_info(self):
                return {"steps": self.steps}

            def observation_for_policy(self, observation, **kwargs):
                return {}

            def step(self, action):
                self.steps += 1
                self.last_executed_action = np.asarray(action)
                observation = {
                    "robot_obs": np.zeros(15),
                    "scene_obs": np.zeros(24),
                    "rgb_obs": {"rgb_static": np.zeros((2, 2, 3), np.uint8)},
                }
                return observation, 0, False, self.get_info()

        class Client:
            def __init__(self):
                self.sessions = []

            def reset(self, **kwargs):
                self.sessions.append(kwargs["episode_id"])

            def step(self, episode_id, observation, step):
                return np.zeros(7), {"replanned": True}

        class Oracle:
            @staticmethod
            def get_task_info_for_set(start, current, tasks):
                return {"open_drawer"} if current["steps"] >= 2 else set()

        adapter = Adapter()
        client = Client()
        engine = SimpleNamespace(
            adapter=adapter,
            client=client,
            task_oracle=Oracle(),
            replan_interval=1,
            base_seed=42,
            restore_thresholds={
                "tcp_position_m": 1,
                "tcp_orientation_rad": 1,
                "gripper_width_m": 1,
                "joint_position_rad": 1,
                "fixed_scene": 1,
                "object_position_m": 1,
                "object_orientation_rad": 1,
            },
            mask_provider=None,
            mask_overlay=None,
            video_dir=None,
            video_fps=30,
        )
        result = CalvinOracleRetryAgent(engine, max_attempts=2).run(Episode())
        self.assertTrue(result["success"])
        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual(adapter.reset_count, 1)
        self.assertEqual(len(client.sessions), 2)
        self.assertEqual(
            set(
                [
                    "attempts",
                    "actions",
                    "raw_actions",
                    "success",
                    "termination_reason",
                    "steps_executed",
                ]
            ).difference(result),
            set(),
        )

    def test_attempt_runner_is_reusable_with_current_scene_and_legacy_verifier(self):
        from types import SimpleNamespace

        class Episode:
            episode_id = "attempt-runner-test"
            task = "open_drawer"
            instruction = "open the drawer"
            max_steps = 1

        class Adapter:
            def __init__(self):
                self.last_executed_action = None

            @staticmethod
            def observation_for_policy(observation, **kwargs):
                return {"state": observation["state"]}

            def step(self, action):
                self.last_executed_action = np.asarray(action).copy()
                observation = {"state": int(action[0])}
                return observation, 0, False, {"state": observation["state"]}

        class Client:
            def __init__(self):
                self.next_state = 1
                self.resets = []

            def reset(self, **kwargs):
                self.resets.append(kwargs)

            def step(self, episode_id, observation, step):
                action = np.zeros(7, dtype=np.float32)
                action[0] = self.next_state
                self.next_state += 1
                return action, {"replanned": True}

        class LegacyVerifier:
            @staticmethod
            def evaluate(current_info):
                return SimpleNamespace(
                    success=current_info["state"] >= 2,
                    status="success" if current_info["state"] >= 2 else "unknown",
                    reason="legacy",
                    confidence=1.0,
                    evidence={},
                )

        adapter = Adapter()
        client = Client()
        engine = SimpleNamespace(
            adapter=adapter,
            client=client,
            replan_interval=4,
            mask_provider=None,
            mask_overlay=None,
            video_dir=None,
        )
        runner = CalvinSkillAttemptRunner(engine, LegacyVerifier())
        first = runner.run(Episode(), {"state": 0}, 1, "attempt-1", 10, 0)
        second = runner.run(
            Episode(), first.observation, 2, "attempt-2", 11, first.steps_executed
        )
        self.assertFalse(first.success)
        self.assertTrue(second.success)
        self.assertEqual(first.observation["state"], 1)
        self.assertEqual(second.success_step, 2)
        self.assertEqual(len(client.resets), 2)

    def test_oracle_verifier_keeps_baseline_and_matches_success_guard(self):
        from types import SimpleNamespace

        baseline = {"drawer": 0}

        class Oracle:
            def __init__(self):
                self.starts = []

            def get_task_info_for_set(self, start, current, tasks):
                self.starts.append(start)
                return {"rotate_red_block_right"}

        oracle = Oracle()
        verifier = CalvinOracleTaskVerifier(oracle)
        verifier.begin(
            SimpleNamespace(
                oracle_task="rotate_red_block_right",
                success_guard={
                    "gripper": "open",
                    "min_width_m": 0.04,
                    "consecutive_steps": 2,
                },
            ),
            baseline,
        )
        observation = {
            "robot_obs": np.asarray([0, 0, 0, 0, 0, 0, 0.05])
        }
        open_action = np.asarray([0, 0, 0, 0, 0, 0, 1])
        first = verifier.evaluate(
            {"drawer": 1}, observation=observation, executed_action=open_action
        )
        second = verifier.evaluate(
            {"drawer": 2}, observation=observation, executed_action=open_action
        )
        self.assertFalse(first.success)
        self.assertEqual(first.reason, "success_guard_not_satisfied")
        self.assertTrue(second.success)
        self.assertEqual(second.evidence["success_guard_streak"], 2)
        self.assertTrue(all(start is baseline for start in oracle.starts))


if __name__ == "__main__":
    unittest.main()
