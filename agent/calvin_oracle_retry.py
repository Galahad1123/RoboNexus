"""CALVIN-oracle retry agent that continues from the current simulator state."""

import time
from pathlib import Path

import numpy as np

from calvin.evaluation.remote_policy.rollout_engine import (
    derive_episode_seed,
    restoration_metrics,
)


class CalvinOracleRetryAgent:
    """Run repeated policy sessions without resetting between failed attempts."""

    def __init__(self, engine, max_attempts):
        self.engine = engine
        self.max_attempts = int(max_attempts)
        if self.max_attempts < 2:
            raise ValueError("Retry agent requires max_attempts >= 2")

    def run(self, episode):
        engine = self.engine
        robot_obs, scene_obs = episode.load_initial_state()
        observation = engine.adapter.reset(robot_obs, scene_obs)
        restore_metrics = restoration_metrics(
            robot_obs,
            scene_obs,
            observation["robot_obs"],
            observation["scene_obs"],
        )
        failed = {
            name: {"value": restore_metrics[name], "threshold": threshold}
            for name, threshold in engine.restore_thresholds.items()
            if restore_metrics[name] > threshold
        }
        if failed:
            raise RuntimeError(
                "Initial-state restoration exceeded component thresholds: {}".format(
                    failed
                )
            )
        if engine.mask_provider is not None:
            engine.mask_provider.reset(episode)

        zero_action = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        for _ in range(episode.settle_steps):
            observation, _, _, _ = engine.adapter.step(zero_action)
        oracle_start_info = engine.adapter.get_info()

        attempts = []
        all_actions = []
        all_raw_actions = []
        all_latencies = []
        all_replan_steps = []
        success = False
        success_step = None
        total_steps = 0
        termination_reason = "retry_exhausted"
        frames = []

        for attempt_index in range(self.max_attempts):
            session_id = "{}__attempt_{:02d}".format(
                episode.episode_id, attempt_index + 1
            )
            seed = derive_episode_seed(
                engine.base_seed,
                "{}:attempt:{}".format(episode.episode_id, attempt_index),
            )
            engine.client.reset(
                episode_id=session_id,
                task=episode.instruction,
                replan_interval=engine.replan_interval,
                episode_seed=seed,
            )
            attempt_start_step = total_steps
            attempt_replans = []
            attempt_reason = "timeout"
            for local_step in range(episode.max_steps):
                role_masks = (
                    engine.mask_provider.masks(observation)
                    if engine.mask_provider is not None
                    else None
                )
                policy_observation = engine.adapter.observation_for_policy(
                    observation,
                    role_masks=role_masks,
                    overlay=engine.mask_overlay,
                    frame_index=total_steps,
                )
                if engine.video_dir:
                    video_frame = (
                        policy_observation["observation.images.static"]
                        if engine.mask_overlay is not None
                        else observation["rgb_obs"]["rgb_static"]
                    )
                    frames.append(np.asarray(video_frame).copy())
                started = time.perf_counter()
                action, metadata = engine.client.step(
                    session_id, policy_observation, local_step
                )
                all_latencies.append((time.perf_counter() - started) * 1000.0)
                all_raw_actions.append(action.tolist())
                if metadata.get("replanned"):
                    attempt_replans.append(local_step)
                    all_replan_steps.append(total_steps)

                observation, _, done, current_info = engine.adapter.step(action)
                all_actions.append(engine.adapter.last_executed_action.tolist())
                total_steps += 1
                achieved = engine.task_oracle.get_task_info_for_set(
                    oracle_start_info, current_info, {episode.oracle_task}
                )
                if episode.oracle_task in achieved:
                    success = True
                    success_step = total_steps
                    attempt_reason = "success"
                    termination_reason = "success"
                    break
                if done:
                    attempt_reason = "environment_done"
                    termination_reason = "environment_done"
                    break

            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "session_id": session_id,
                    "episode_seed": seed,
                    "start_step": attempt_start_step,
                    "steps_executed": total_steps - attempt_start_step,
                    "replan_steps": attempt_replans,
                    "termination_reason": attempt_reason,
                    "success": success,
                }
            )
            if success or attempt_reason == "environment_done":
                break

        video_path = None
        if engine.video_dir and frames:
            import imageio

            suffix = "success" if success else "failure"
            video_path = Path(engine.video_dir) / "{}__{}__retry_{}__{}.mp4".format(
                episode.task,
                episode.episode_id,
                len(attempts),
                suffix,
            )
            imageio.mimwrite(
                str(video_path),
                frames,
                fps=engine.video_fps,
                macro_block_size=8,
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
            "steps_executed": total_steps,
            "max_steps": episode.max_steps,
            "max_attempts": self.max_attempts,
            "attempt_count": len(attempts),
            "attempts": attempts,
            "termination_reason": termination_reason,
            "replan_interval": engine.replan_interval,
            "episode_seed": attempts[0]["episode_seed"],
            "replan_steps": all_replan_steps,
            "inference_requests": len(all_replan_steps),
            "mean_request_latency_ms": (
                float(np.mean(all_latencies)) if all_latencies else None
            ),
            "max_request_latency_ms": (
                float(np.max(all_latencies)) if all_latencies else None
            ),
            "restore_metrics": restore_metrics,
            "raw_actions": all_raw_actions,
            "actions": all_actions,
            "video": str(video_path) if video_path else None,
        }
