"""Generic single-task CALVIN rollout loop."""

import time
from pathlib import Path

import numpy as np


DEFAULT_RESTORE_THRESHOLDS = {
    "tcp_position_m": 0.005,
    "tcp_orientation_rad": 0.02,
    "gripper_width_m": 0.005,
    "joint_position_rad": 0.01,
    "fixed_scene": 0.02,
    "object_position_m": 0.01,
    "object_orientation_rad": 0.15,
}


def _rotation_error(euler_a, euler_b):
    from scipy.spatial.transform import Rotation

    start = Rotation.from_euler("xyz", euler_a)
    restored = Rotation.from_euler("xyz", euler_b)
    return float((restored * start.inv()).magnitude())


def restoration_metrics(expected_robot, expected_scene, restored_robot, restored_scene):
    """Compare physical state components while ignoring discrete command history.

    CALVIN robot_obs[14] is the last commanded gripper direction, not a
    physical state. Euler angles also need SO(3)-aware comparison because
    equivalent angles can differ by approximately 2*pi.
    """
    expected_robot = np.asarray(expected_robot)
    restored_robot = np.asarray(restored_robot)
    expected_scene = np.asarray(expected_scene)
    restored_scene = np.asarray(restored_scene)
    if expected_robot.shape != (15,) or restored_robot.shape != (15,):
        raise ValueError("Expected 15-D CALVIN robot observations")
    if expected_scene.shape != restored_scene.shape or expected_scene.shape != (24,):
        raise ValueError("Expected matching 24-D CALVIN scene observations")

    object_errors_position = []
    object_errors_orientation = []
    # task-D scene state: 6 fixed scalars followed by three xyz+rpy objects.
    for offset in range(6, 24, 6):
        object_errors_position.append(
            float(
                np.linalg.norm(
                    restored_scene[offset : offset + 3]
                    - expected_scene[offset : offset + 3]
                )
            )
        )
        object_errors_orientation.append(
            _rotation_error(
                expected_scene[offset + 3 : offset + 6],
                restored_scene[offset + 3 : offset + 6],
            )
        )

    return {
        "tcp_position_m": float(
            np.linalg.norm(restored_robot[:3] - expected_robot[:3])
        ),
        "tcp_orientation_rad": _rotation_error(
            expected_robot[3:6], restored_robot[3:6]
        ),
        "gripper_width_m": float(abs(restored_robot[6] - expected_robot[6])),
        "joint_position_rad": float(
            np.max(
                np.abs(
                    np.arctan2(
                        np.sin(restored_robot[7:14] - expected_robot[7:14]),
                        np.cos(restored_robot[7:14] - expected_robot[7:14]),
                    )
                )
            )
        ),
        "fixed_scene": float(np.max(np.abs(restored_scene[:6] - expected_scene[:6]))),
        "object_position_m": max(object_errors_position),
        "object_orientation_rad": max(object_errors_orientation),
        "gripper_command_expected": int(expected_robot[14]),
        "gripper_command_restored": int(restored_robot[14]),
    }


class RolloutEngine:
    def __init__(
        self,
        adapter,
        client,
        task_oracle,
        replan_interval,
        video_dir=None,
        video_fps=30,
        restore_thresholds=None,
    ):
        self.adapter = adapter
        self.client = client
        self.task_oracle = task_oracle
        self.replan_interval = int(replan_interval)
        self.video_dir = Path(video_dir) if video_dir else None
        self.video_fps = int(video_fps)
        self.restore_thresholds = dict(
            restore_thresholds or DEFAULT_RESTORE_THRESHOLDS
        )
        if self.video_dir:
            self.video_dir.mkdir(parents=True, exist_ok=True)

    def run(self, episode):
        robot_obs, scene_obs = episode.load_initial_state()
        observation = self.adapter.reset(robot_obs, scene_obs)
        restore_metrics = restoration_metrics(
            robot_obs,
            scene_obs,
            observation["robot_obs"],
            observation["scene_obs"],
        )
        failed_restore_metrics = {
            name: {
                "value": restore_metrics[name],
                "threshold": threshold,
            }
            for name, threshold in self.restore_thresholds.items()
            if restore_metrics[name] > threshold
        }
        if failed_restore_metrics:
            raise RuntimeError(
                "Initial-state restoration exceeded component thresholds: {}".format(
                    failed_restore_metrics
                )
            )

        start_info = self.adapter.get_info()
        self.client.reset(
            episode_id=episode.episode_id,
            task=episode.instruction,
            replan_interval=self.replan_interval,
        )

        zero_action = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        for _ in range(episode.settle_steps):
            observation, _, _, _ = self.adapter.step(zero_action)
        # Oracle starts after optional settling, so settling itself cannot count
        # as policy success.
        start_info = self.adapter.get_info()

        frames = []
        actions = []
        raw_actions = []
        latencies = []
        replan_steps = []
        success = False
        termination_reason = "timeout"
        success_step = None

        for step in range(episode.max_steps):
            if self.video_dir:
                frames.append(np.asarray(observation["rgb_obs"]["rgb_static"]).copy())
            policy_observation = self.adapter.observation_for_policy(observation)
            started = time.perf_counter()
            action, metadata = self.client.step(
                episode.episode_id, policy_observation, step
            )
            latencies.append((time.perf_counter() - started) * 1000.0)
            raw_actions.append(action.tolist())
            if metadata.get("replanned"):
                replan_steps.append(step)

            observation, _, done, current_info = self.adapter.step(action)
            actions.append(self.adapter.last_executed_action.tolist())
            achieved = self.task_oracle.get_task_info_for_set(
                start_info, current_info, {episode.oracle_task}
            )
            if episode.oracle_task in achieved:
                success = True
                success_step = step + 1
                termination_reason = "success"
                break
            if done:
                termination_reason = "environment_done"
                break

        video_path = None
        if self.video_dir and frames:
            import imageio

            suffix = "success" if success else "failure"
            video_path = self.video_dir / "{}_{}.mp4".format(episode.episode_id, suffix)
            imageio.mimwrite(
                str(video_path),
                frames,
                fps=self.video_fps,
                # 200 is divisible by 8, so retain the simulator resolution
                # instead of imageio's default 208x208 macroblock padding.
                macro_block_size=8,
                # swscale alignment messages only concern encoding speed and
                # otherwise flood long evaluation logs.
                ffmpeg_log_level="error",
            )

        return {
            "episode_id": episode.episode_id,
            "task": episode.task,
            "instruction": episode.instruction,
            "oracle_task": episode.oracle_task,
            "source_frame": str(episode.source_frame),
            "success": success,
            "success_step": success_step,
            "steps_executed": len(actions),
            "max_steps": episode.max_steps,
            "termination_reason": termination_reason,
            "replan_interval": self.replan_interval,
            "replan_steps": replan_steps,
            "inference_requests": len(replan_steps),
            "mean_request_latency_ms": float(np.mean(latencies)) if latencies else None,
            "max_request_latency_ms": float(np.max(latencies)) if latencies else None,
            "restore_metrics": restore_metrics,
            "raw_actions": raw_actions,
            "actions": actions,
            "video": str(video_path) if video_path else None,
        }
