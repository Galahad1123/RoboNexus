"""Legacy one-attempt policy session for a routed canonical atomic skill.

Resetting an attempt resets only the remote policy's temporal state. The
environment remains owned by the backend and is deliberately not reset here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.logging import NullEventLogger

from .schemas import AtomicAttemptContext, AtomicAttemptSummary, PolicyStepResult


class AtomicSkillExecutionError(RuntimeError):
    pass


class AtomicSkillExecutor:
    """Route one skill and expose one policy action at a time."""

    def __init__(self, router: Any, clients: Any, events: Any | None = None) -> None:
        self.router = router
        self.clients = clients
        self.events = events or NullEventLogger()
        self._active: AtomicAttemptContext | None = None
        self._active_client: Any = None
        self._last_step_index: int | None = None

    @property
    def active_attempt(self) -> AtomicAttemptContext | None:
        return self._active

    def begin_attempt(
        self,
        skill_id: str,
        execution_id: str,
        attempt: int,
        episode_seed: int,
    ) -> AtomicAttemptContext:
        if self._active is not None:
            raise AtomicSkillExecutionError(
                "Attempt {!r} is still active; end it before starting another".format(
                    self._active.policy_episode_id
                )
            )
        if not isinstance(execution_id, str) or not execution_id.strip():
            raise ValueError("execution_id must be a non-empty string")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        if (
            isinstance(episode_seed, bool)
            or not isinstance(episode_seed, int)
            or episode_seed < 0
        ):
            raise ValueError("episode_seed must be a non-negative integer")

        decision = self.router.route(skill_id)
        skill = decision.skill
        policy = decision.policy
        client = self.clients.get(policy.policy_id)
        policy_episode_id = "{}:{}:attempt-{}".format(
            execution_id, skill.skill_id, attempt
        )
        response = client.reset(
            policy_episode_id,
            skill.instruction,
            skill.replan_interval,
            episode_seed,
        )
        if response.get("type") != "reset_ack" or response.get("ok") is not True:
            raise AtomicSkillExecutionError(
                "Policy {!r} rejected attempt reset: {!r}".format(
                    policy.policy_id, response
                )
            )
        if response.get("episode_seed") != episode_seed:
            raise AtomicSkillExecutionError(
                "Policy {!r} acknowledged episode_seed {!r}, expected {!r}".format(
                    policy.policy_id, response.get("episode_seed"), episode_seed
                )
            )
        if response.get("replan_interval") != skill.replan_interval:
            raise AtomicSkillExecutionError(
                "Policy {!r} acknowledged replan_interval {!r}, expected {!r}".format(
                    policy.policy_id,
                    response.get("replan_interval"),
                    skill.replan_interval,
                )
            )

        context = AtomicAttemptContext(
            execution_id=execution_id,
            policy_episode_id=policy_episode_id,
            skill_id=skill.skill_id,
            policy_id=policy.policy_id,
            instruction=skill.instruction,
            attempt=attempt,
            episode_seed=episode_seed,
            replan_interval=skill.replan_interval,
        )
        self._active = context
        self._active_client = client
        self._last_step_index = None
        self.events.emit(
            "atomic_attempt_started",
            execution_id=execution_id,
            policy_episode_id=policy_episode_id,
            skill_id=skill.skill_id,
            policy_id=policy.policy_id,
            endpoint=policy.endpoint,
            attempt=attempt,
            episode_seed=episode_seed,
            replan_interval=skill.replan_interval,
            scene_continuation=True,
        )
        return context

    def step(self, observation: Any, step_index: int) -> PolicyStepResult:
        context = self._require_active()
        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or step_index < 0
        ):
            raise ValueError("step_index must be a non-negative integer")
        if self._last_step_index is not None and step_index <= self._last_step_index:
            raise AtomicSkillExecutionError(
                "step_index must increase within an attempt; last={}, got={}".format(
                    self._last_step_index, step_index
                )
            )
        response = self._active_client.step(
            context.policy_episode_id, observation, step_index
        )
        if response.get("type") != "action" or "action" not in response:
            raise AtomicSkillExecutionError(
                "Policy {!r} returned an invalid action response: {!r}".format(
                    context.policy_id, response
                )
            )
        action = np.asarray(response["action"], dtype=np.float32).reshape(-1).copy()
        expected_dim = int(self._active_client.metadata["action_dim"])
        if action.size != expected_dim:
            raise AtomicSkillExecutionError(
                "Policy {!r} returned action dim {}, expected {}".format(
                    context.policy_id, action.size, expected_dim
                )
            )
        metadata = response.get("metadata", {})
        if not isinstance(metadata, dict):
            raise AtomicSkillExecutionError("action metadata must be a mapping")
        self._last_step_index = step_index
        return PolicyStepResult(
            action=action,
            policy_id=context.policy_id,
            skill_id=context.skill_id,
            attempt=context.attempt,
            step_index=step_index,
            metadata=dict(metadata),
        )

    def end_attempt(
        self, success: bool, steps: int, termination_reason: str
    ) -> AtomicAttemptSummary:
        context = self._require_active()
        if not isinstance(success, bool):
            raise TypeError("success must be boolean")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        if not isinstance(termination_reason, str) or not termination_reason.strip():
            raise ValueError("termination_reason must be a non-empty string")
        summary = AtomicAttemptSummary(
            context=context,
            success=success,
            steps=steps,
            termination_reason=termination_reason,
        )
        self.events.emit(
            "atomic_attempt_succeeded" if success else "atomic_attempt_failed",
            execution_id=context.execution_id,
            policy_episode_id=context.policy_episode_id,
            skill_id=context.skill_id,
            policy_id=context.policy_id,
            attempt=context.attempt,
            steps=steps,
            reason=termination_reason,
        )
        self._active = None
        self._active_client = None
        self._last_step_index = None
        return summary

    def _require_active(self) -> AtomicAttemptContext:
        if self._active is None:
            raise AtomicSkillExecutionError("No atomic skill attempt is active")
        return self._active
