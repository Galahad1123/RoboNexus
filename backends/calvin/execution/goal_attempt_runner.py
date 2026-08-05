"""Bridge the generic goal state machine to routed CALVIN policy attempts."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.legacy.execution.schemas import AttemptResult

from .attempt_runner import CalvinSkillAttemptRunner
from ..verification.task_verifier import CalvinOracleTaskVerifier


class CalvinGoalAttemptRunner:
    """Select a DiT per skill and execute one attempt in the current scene."""

    def __init__(
        self,
        adapter,
        task_oracle,
        skill_registry,
        policy_router,
        client_pool,
        event_logger=None,
        mask_provider=None,
        mask_overlay=None,
        mask_roles_by_skill=None,
        video_dir=None,
        video_fps=30,
        max_steps_override=None,
    ):
        self.adapter = adapter
        self.task_oracle = task_oracle
        self.skill_registry = skill_registry
        self.policy_router = policy_router
        self.client_pool = client_pool
        self.events = event_logger
        self.mask_provider = mask_provider
        self.mask_overlay = mask_overlay
        self.mask_roles_by_skill = dict(mask_roles_by_skill or {})
        self.video_dir = video_dir
        self.video_fps = int(video_fps)
        self.max_steps_override = max_steps_override
        self._sessions = {}

    def run_attempt(self, request):
        key = (request.goal.goal_id, request.plan_step)
        session = self._sessions.get(key)
        if session is None:
            session = self._create_session(request)
            self._sessions[key] = session
        elif session.skill_id != request.skill_id:
            raise RuntimeError(
                "plan_step {} changed skill from {!r} to {!r}".format(
                    request.plan_step, session.skill_id, request.skill_id
                )
            )

        session_id = "{}__step_{:02d}__replan_{:02d}__attempt_{:02d}".format(
            request.goal.goal_id,
            request.plan_step,
            request.replans_used,
            request.attempt,
        )
        result = session.runner.run(
            episode=session.episode,
            observation=request.observation,
            attempt=request.attempt,
            session_id=session_id,
            episode_seed=request.episode_seed,
            global_start_step=0,
        )
        video_path = self._write_attempt_video(
            result.frames, session_id, result.success
        )
        return AttemptResult(
            success=result.success,
            steps=result.steps_executed,
            termination_reason=result.termination_reason,
            observation=result.observation,
            metadata={
                "policy_id": session.policy_id,
                "session_id": session_id,
                "success_step": result.success_step,
                "environment_done": result.environment_done,
                "raw_actions": result.raw_actions,
                "actions": result.actions,
                "latencies_ms": result.latencies_ms,
                "replan_steps": result.replan_steps,
                "video": str(video_path) if video_path is not None else None,
            },
        )

    def _write_attempt_video(self, frames, session_id, success):
        if not self.video_dir or not frames:
            return None

        import imageio

        video_dir = Path(self.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        suffix = "success" if success else "failure"
        video_path = video_dir / "{}__{}.mp4".format(session_id, suffix)
        imageio.mimwrite(
            str(video_path),
            frames,
            fps=self.video_fps,
            macro_block_size=8,
            ffmpeg_log_level="error",
        )
        return video_path

    def _create_session(self, request):
        decision = self.policy_router.route(request.skill_id)
        skill = decision.skill
        policy = decision.policy
        client = _PolicyResponseAdapter(self.client_pool.get(policy.policy_id))
        max_steps = (
            int(self.max_steps_override)
            if self.max_steps_override is not None
            else int(skill.max_steps)
        )
        if max_steps < 1:
            raise ValueError("max_steps_override must be >= 1")
        episode = SimpleNamespace(
            episode_id="{}__step_{:02d}".format(
                request.goal.goal_id, request.plan_step
            ),
            task=skill.skill_id,
            instruction=skill.instruction,
            oracle_task=skill.oracle_task,
            max_steps=max_steps,
            success_guard=skill.success_guard,
            mask_roles=self.mask_roles_by_skill.get(skill.skill_id),
        )
        if self.mask_provider is not None:
            if not episode.mask_roles:
                raise RuntimeError(
                    "Mask provider is enabled but skill {!r} has no mask_roles "
                    "for the current scene".format(skill.skill_id)
                )
            self.mask_provider.reset(episode)
        verifier = CalvinOracleTaskVerifier(self.task_oracle)
        verifier.begin(episode, self.adapter.get_info())
        engine = SimpleNamespace(
            adapter=self.adapter,
            client=client,
            replan_interval=skill.replan_interval,
            mask_provider=self.mask_provider,
            mask_overlay=self.mask_overlay,
            video_dir=self.video_dir,
            video_fps=self.video_fps,
        )
        return SimpleNamespace(
            skill_id=skill.skill_id,
            policy_id=policy.policy_id,
            episode=episode,
            runner=CalvinSkillAttemptRunner(engine, verifier, self.events),
        )


class _PolicyResponseAdapter:
    """Adapt core PolicyEndpointClient responses to the legacy CALVIN client API."""

    def __init__(self, client):
        self.client = client

    def reset(self, **kwargs):
        return self.client.reset(**kwargs)

    def step(self, episode_id, observation, step):
        response = self.client.step(episode_id, observation, step)
        action = np.asarray(response["action"], dtype=np.float32).reshape(-1).copy()
        return action, dict(response.get("metadata", {}))
