"""Generic single-task CALVIN rollout loop."""

import hashlib
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


def derive_episode_seed(base_seed, episode_id):
    """Create a stable PyTorch-compatible seed independent of process hash state."""
    digest = hashlib.sha256(
        "{}:{}".format(int(base_seed), str(episode_id)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**63 - 1)


def success_guard_satisfied(guard, observation, executed_action):
    if not guard:
        return True
    gripper = guard.get("gripper")
    if gripper is None:
        return True
    width = float(observation["robot_obs"][6])
    command = float(executed_action[-1])
    if gripper == "open":
        return command > 0 and width >= float(guard.get("min_width_m", 0.04))
    if gripper == "closed":
        return command < 0 and width <= float(guard.get("max_width_m", 0.04))
    raise ValueError("Unsupported success-guard gripper state: {}".format(gripper))


class RolloutEngine:
    def __init__(
        self,
        adapter,
        client,
        task_oracle,
        replan_interval,
        base_seed=42,
        video_dir=None,
        video_fps=30,
        restore_thresholds=None,
        mask_provider=None,
        mask_overlay=None,
    ):
        self.adapter = adapter
        self.client = client
        self.task_oracle = task_oracle
        self.replan_interval = int(replan_interval)
        self.base_seed = int(base_seed)
        self.video_dir = Path(video_dir) if video_dir else None
        self.video_fps = int(video_fps)
        self.restore_thresholds = dict(
            restore_thresholds or DEFAULT_RESTORE_THRESHOLDS
        )
        self.mask_provider = mask_provider
        self.mask_overlay = mask_overlay
        if self.video_dir:
            self.video_dir.mkdir(parents=True, exist_ok=True)

    def run(self, episode):
        robot_obs, scene_obs = episode.load_initial_state()
        observation = self.adapter.reset(robot_obs, scene_obs)
        if self.mask_provider is not None:
            self.mask_provider.reset(episode)
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
        episode_seed = derive_episode_seed(self.base_seed, episode.episode_id)
        self.client.reset(
            episode_id=episode.episode_id,
            task=episode.instruction,
            replan_interval=self.replan_interval,
            episode_seed=episode_seed,
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
        success_guard_streak = 0

        for step in range(episode.max_steps):
            role_masks = (
                self.mask_provider.masks(observation)
                if self.mask_provider is not None
                else None
            )
            policy_observation = self.adapter.observation_for_policy(
                observation,
                role_masks=role_masks,
                overlay=self.mask_overlay,
                frame_index=step,
            )
            if self.video_dir:
                video_frame = (
                    policy_observation["observation.images.static"]
                    if self.mask_overlay is not None
                    else observation["rgb_obs"]["rgb_static"]
                )
                frames.append(np.asarray(video_frame).copy())
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
            success_guard = getattr(episode, "success_guard", None)
            guarded_success = episode.oracle_task in achieved and success_guard_satisfied(
                success_guard,
                observation,
                self.adapter.last_executed_action,
            )
            success_guard_streak = success_guard_streak + 1 if guarded_success else 0
            required_success_steps = int(
                (success_guard or {}).get("consecutive_steps", 1)
            )
            if success_guard_streak >= required_success_steps:
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
            video_path = self.video_dir / "{}__{}__{}.mp4".format(
                episode.task, episode.episode_id, suffix
            )
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
            "episode_variant": episode.episode_variant,
            "instruction": episode.instruction,
            "oracle_task": episode.oracle_task,
            "success_guard": getattr(episode, "success_guard", None),
            "source_frame": str(episode.source_frame),
            "success": success,
            "success_step": success_step,
            "steps_executed": len(actions),
            "max_steps": episode.max_steps,
            "termination_reason": termination_reason,
            "replan_interval": self.replan_interval,
            "episode_seed": episode_seed,
            "replan_steps": replan_steps,
            "inference_requests": len(replan_steps),
            "mean_request_latency_ms": float(np.mean(latencies)) if latencies else None,
            "max_request_latency_ms": float(np.max(latencies)) if latencies else None,
            "restore_metrics": restore_metrics,
            "raw_actions": raw_actions,
            "actions": actions,
            "video": str(video_path) if video_path else None,
        }
