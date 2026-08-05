"""CALVIN retry runner that continues from the current simulator state."""

from pathlib import Path

import numpy as np

from backends.calvin.event_logger import NullEventLogger
from backends.calvin.attempt_runner import CalvinSkillAttemptRunner
from backends.calvin.task_verifier import CalvinOracleTaskVerifier
from calvin.evaluation.remote_policy.rollout_engine import (
    derive_episode_seed,
    restoration_metrics,
)


class CalvinOracleRetryAgent:
    """Run repeated policy sessions without resetting between failed attempts."""

    def __init__(
        self,
        engine,
        max_attempts,
        recovery=None,
        event_logger=None,
        verifier=None,
        attempt_runner=None,
    ):
        self.engine = engine
        self.max_attempts = int(max_attempts)
        self.recovery = recovery
        self.events = event_logger or NullEventLogger()
        self.verifier = verifier or CalvinOracleTaskVerifier(engine.task_oracle)
        self.attempt_runner = attempt_runner or CalvinSkillAttemptRunner(
            engine, self.verifier, self.events
        )
        if self.max_attempts < 2:
            raise ValueError("Retry agent requires max_attempts >= 2")

    def run(self, episode):
        engine = self.engine
        retry_config = getattr(episode, "retry", {}) or {}
        retry_enabled = bool(retry_config.get("enabled", True))
        effective_max_attempts = self.max_attempts if retry_enabled else 1
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
        self.verifier.begin(episode, oracle_start_info)

        attempts = []
        all_actions = []
        all_raw_actions = []
        all_latencies = []
        all_replan_steps = []
        success = False
        success_step = None
        total_steps = 0
        recovery_steps = 0
        recovery_actions = []
        termination_reason = "retry_exhausted"
        frames = []
        self.events.emit(
            "episode_started",
            episode_id=episode.episode_id,
            task=episode.task,
            max_attempts=effective_max_attempts,
            retry_enabled=retry_enabled,
        )

        for attempt_index in range(effective_max_attempts):
            recovery_record = None
            if attempt_index > 0:
                if self.recovery is None:
                    self.events.emit(
                        "recovery_skipped",
                        episode_id=episode.episode_id,
                        task=episode.task,
                        attempt=attempt_index + 1,
                        reason="no_recovery_backend",
                    )
                else:
                    self.events.emit(
                        "recovery_started",
                        episode_id=episode.episode_id,
                        task=episode.task,
                        attempt=attempt_index + 1,
                        strategy=retry_config.get("strategy", "oracle_pre_approach"),
                    )
                    try:
                        recovery_result = self.recovery.recover(episode, observation)
                    except Exception as exc:
                        termination_reason = "recovery_exception"
                        self.events.emit(
                            "recovery_failed",
                            episode_id=episode.episode_id,
                            task=episode.task,
                            attempt=attempt_index + 1,
                            reason=termination_reason,
                            error="{}: {}".format(type(exc).__name__, exc),
                        )
                        break
                    observation = recovery_result.observation
                    recovery_record = recovery_result.as_dict()
                    recovery_steps += recovery_result.steps
                    recovery_actions.extend(recovery_result.actions or [])
                    self.events.emit(
                        "recovery_completed"
                        if recovery_result.success
                        else "recovery_failed",
                        episode_id=episode.episode_id,
                        task=episode.task,
                        attempt=attempt_index + 1,
                        **recovery_record
                    )
                    if not recovery_result.success:
                        termination_reason = "recovery_failed"
                        break
            session_id = "{}__attempt_{:02d}".format(
                episode.episode_id, attempt_index + 1
            )
            seed = derive_episode_seed(
                engine.base_seed,
                "{}:attempt:{}".format(episode.episode_id, attempt_index),
            )
            attempt_start_step = total_steps
            self.events.emit(
                "attempt_started",
                episode_id=episode.episode_id,
                task=episode.task,
                attempt=attempt_index + 1,
                session_id=session_id,
                episode_seed=seed,
            )
            attempt_result = self.attempt_runner.run(
                episode=episode,
                observation=observation,
                attempt=attempt_index + 1,
                session_id=session_id,
                episode_seed=seed,
                global_start_step=total_steps,
            )
            observation = attempt_result.observation
            attempt_reason = attempt_result.termination_reason
            attempt_replans = attempt_result.replan_steps
            total_steps += attempt_result.steps_executed
            all_latencies.extend(attempt_result.latencies_ms)
            all_raw_actions.extend(attempt_result.raw_actions)
            all_actions.extend(attempt_result.actions)
            all_replan_steps.extend(attempt_result.global_replan_steps)
            frames.extend(attempt_result.frames)
            if attempt_result.success:
                success = True
                success_step = attempt_result.success_step
                termination_reason = "success"
            elif attempt_result.environment_done:
                termination_reason = "environment_done"

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
                    "recovery": recovery_record,
                }
            )
            self.events.emit(
                "attempt_succeeded" if success else "attempt_failed",
                episode_id=episode.episode_id,
                task=episode.task,
                attempt=attempt_index + 1,
                steps_executed=total_steps - attempt_start_step,
                reason=attempt_reason,
            )
            if success or attempt_result.environment_done:
                break

        if (
            not success
            and not retry_enabled
            and termination_reason == "retry_exhausted"
        ):
            termination_reason = "retry_disabled"

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

        record = {
            "episode_id": episode.episode_id,
            "task": episode.task,
            "instruction": episode.instruction,
            "oracle_task": episode.oracle_task,
            "source_frame": str(episode.source_frame),
            "success": success,
            "success_step": success_step,
            "steps_executed": total_steps,
            "policy_steps": total_steps,
            "recovery_steps": recovery_steps,
            "total_environment_steps": total_steps + recovery_steps,
            "max_steps": episode.max_steps,
            "max_attempts": effective_max_attempts,
            "retry_enabled": retry_enabled,
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
            "recovery_actions": recovery_actions,
            "video": str(video_path) if video_path else None,
        }
        self.events.emit(
            "episode_finished",
            episode_id=episode.episode_id,
            task=episode.task,
            success=success,
            attempt_count=len(attempts),
            policy_steps=total_steps,
            recovery_steps=recovery_steps,
            termination_reason=termination_reason,
        )
        return record
