"""Reusable, in-place CALVIN action loop for one policy attempt."""

import inspect
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CalvinAttemptResult:
    observation: object
    success: bool
    termination_reason: str
    steps_executed: int
    success_step: object = None
    environment_done: bool = False
    raw_actions: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    latencies_ms: list = field(default_factory=list)
    replan_steps: list = field(default_factory=list)
    global_replan_steps: list = field(default_factory=list)
    frames: list = field(default_factory=list)


class CalvinSkillAttemptRunner:
    """Execute one atomic skill without resetting the CALVIN environment."""

    def __init__(self, engine, verifier, event_logger=None):
        self.engine = engine
        self.verifier = verifier
        self.events = event_logger
        self._verifier_accepts_context = _accepts_verification_context(verifier)

    def run(
        self,
        episode,
        observation,
        attempt,
        session_id,
        episode_seed,
        global_start_step=0,
    ):
        engine = self.engine
        engine.client.reset(
            episode_id=session_id,
            task=episode.instruction,
            replan_interval=engine.replan_interval,
            episode_seed=episode_seed,
        )

        raw_actions = []
        actions = []
        latencies = []
        replan_steps = []
        global_replan_steps = []
        frames = []
        success = False
        environment_done = False
        termination_reason = "timeout"
        success_step = None

        for local_step in range(episode.max_steps):
            global_step = global_start_step + local_step
            role_masks = (
                engine.mask_provider.masks(observation)
                if engine.mask_provider is not None
                else None
            )
            policy_observation = engine.adapter.observation_for_policy(
                observation,
                role_masks=role_masks,
                overlay=engine.mask_overlay,
                frame_index=global_step,
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
            latencies.append((time.perf_counter() - started) * 1000.0)
            raw_actions.append(np.asarray(action).tolist())
            if metadata.get("replanned"):
                replan_steps.append(local_step)
                global_replan_steps.append(global_step)

            observation, _, done, current_info = engine.adapter.step(action)
            actions.append(engine.adapter.last_executed_action.tolist())
            verification = self._evaluate(
                current_info,
                observation,
                engine.adapter.last_executed_action,
            )
            if verification.success:
                success = True
                success_step = global_step + 1
                termination_reason = "success"
                if self.events is not None:
                    self.events.emit(
                        "verification_result",
                        episode_id=episode.episode_id,
                        task=episode.task,
                        attempt=attempt,
                        status=verification.status,
                        reason=verification.reason,
                        confidence=verification.confidence,
                        evidence=verification.evidence,
                    )
                break
            if done:
                environment_done = True
                termination_reason = "environment_done"
                break

        return CalvinAttemptResult(
            observation=observation,
            success=success,
            termination_reason=termination_reason,
            steps_executed=len(actions),
            success_step=success_step,
            environment_done=environment_done,
            raw_actions=raw_actions,
            actions=actions,
            latencies_ms=latencies,
            replan_steps=replan_steps,
            global_replan_steps=global_replan_steps,
            frames=frames,
        )

    def _evaluate(self, current_info, observation, executed_action):
        if self._verifier_accepts_context:
            return self.verifier.evaluate(
                current_info,
                observation=observation,
                executed_action=executed_action,
            )
        return self.verifier.evaluate(current_info)


def _accepts_verification_context(verifier):
    try:
        parameters = inspect.signature(verifier.evaluate).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or parameter.name == "observation"
        for parameter in parameters
    )
