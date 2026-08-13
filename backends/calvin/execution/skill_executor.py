"""CALVIN implementation of one routed, outlined atomic skill attempt."""

from types import SimpleNamespace

import numpy as np

from calvin.evaluation.remote_policy.masking.gt import CalvinGTMaskProvider
from calvin.evaluation.remote_policy.rollout_engine import derive_episode_seed
from core.domain import GroundingRepresentation, SkillExecutionResult
from core.policies import PolicyConnectionError, PolicyRoutingError
from core.ports import (
    SkillExecutorPort,
    SkillExecutorPortError,
    SkillExecutorUnavailable,
    SkillPreconditionRejected,
)

from ..environment.scene_observer import CalvinSceneObserver
from ..grounding.semantic_vocabulary import load_semantic_vocabulary
from ..verification.task_verifier import CalvinOracleTaskVerifier
from ..verification.state_inspector import CalvinStateInspector
from .attempt_runner import CalvinSkillAttemptRunner
from .outline_styles import load_outline_styles


STATEFUL_RETRY_SKILLS = frozenset(
    {"place_in_drawer", "place_in_slider", "stack_block"}
)


class CalvinSkillExecutor(SkillExecutorPort):
    """Route to a DiT and execute once without resetting the CALVIN scene."""

    def __init__(
        self,
        adapter,
        task_oracle,
        policy_router,
        client_pool,
        vocabulary=None,
        vocabulary_path=None,
        outline_styles=None,
        outline_styles_path=None,
        event_logger=None,
        state_inspector=None,
        video_fps=30,
        environment_episode_id=None,
        base_seed=42,
        initial_robot_observation=None,
    ):
        self.adapter = adapter
        self.task_oracle = task_oracle
        self.policy_router = policy_router
        self.client_pool = client_pool
        self.vocabulary = vocabulary or load_semantic_vocabulary(vocabulary_path)
        self.outline_styles = outline_styles or load_outline_styles(
            outline_styles_path
        )
        self.events = event_logger
        self.state_inspector = state_inspector or CalvinStateInspector()
        self.video_fps = int(video_fps)
        self.environment_episode_id = environment_episode_id
        self.base_seed = int(base_seed)
        self.initial_robot_observation = (
            None
            if initial_robot_observation is None
            else np.asarray(initial_robot_observation, dtype=np.float64).copy()
        )
        if (
            self.initial_robot_observation is not None
            and self.initial_robot_observation.shape != (15,)
        ):
            raise ValueError("initial_robot_observation must be a 15-D vector")
        self._available = {}

    def check_preconditions(self, skill_id):
        inspection = self.state_inspector.inspect(
            skill_id, self.adapter.current_observation
        )
        status = inspection.status.value
        if status in {"invalid_precondition", "satisfied"}:
            raise SkillPreconditionRejected(inspection.reason, inspection.evidence)
        if status == "unknown":
            raise SkillPreconditionRejected(inspection.reason, inspection.evidence)
        return {"status": status, "reason": inspection.reason}

    def check_available(self, skill_id):
        try:
            decision = self.policy_router.route(skill_id)
            client = self.client_pool.get(decision.policy.policy_id)
            response = client.health()
            if response.get("type") != "health" or response.get("ok") is not True:
                raise PolicyConnectionError(
                    "policy {!r} failed health check: {!r}".format(
                        decision.policy.policy_id, response
                    )
                )
        except (PolicyRoutingError, PolicyConnectionError, KeyError) as exc:
            raise SkillExecutorUnavailable(str(exc))
        self._available[skill_id] = (decision, client)
        return decision.policy.policy_id

    def execute(
        self, skill_id, groundings, attempt, logical_subtask_id, step_limit
    ):
        available = self._available.pop(skill_id, None)
        if available is None:
            self.check_available(skill_id)
            available = self._available.pop(skill_id)
        decision, client = available
        try:
            recovery = self._restore_robot_for_retry(skill_id, attempt)
            mask_roles = self._mask_roles(groundings)
            mask_provider = CalvinGTMaskProvider(self.adapter.env)
            episode = SimpleNamespace(
                episode_id=logical_subtask_id,
                task=decision.skill.skill_id,
                instruction=decision.skill.instruction,
                oracle_task=decision.skill.oracle_task,
                max_steps=int(step_limit),
                success_guard=decision.skill.success_guard,
                mask_roles=mask_roles,
            )
            mask_provider.reset(episode)
            try:
                overlay = self.outline_styles[decision.policy.policy_id]
            except KeyError:
                raise SkillExecutorPortError(
                    "no outline style for policy {!r}".format(
                        decision.policy.policy_id
                    )
                )
            verifier = CalvinOracleTaskVerifier(self.task_oracle)
            verifier.begin(episode, self.adapter.get_info())
            engine = SimpleNamespace(
                adapter=self.adapter,
                client=_PolicyResponseAdapter(client),
                replan_interval=decision.skill.replan_interval,
                mask_provider=mask_provider,
                mask_overlay=overlay,
                video_dir=True,
                video_fps=self.video_fps,
            )
            runner = CalvinSkillAttemptRunner(engine, verifier, self.events)
            policy_session_id = "{}__attempt_{:02d}".format(
                logical_subtask_id, attempt
            )
            seed_key = self.environment_episode_id or logical_subtask_id
            if attempt > 1:
                seed_key = "{}:retry:{}".format(seed_key, attempt)
            seed = derive_episode_seed(self.base_seed, seed_key)
            result = runner.run(
                episode=episode,
                observation=self.adapter.current_observation,
                attempt=attempt,
                session_id=policy_session_id,
                episode_seed=seed,
                global_start_step=self.adapter.environment_steps,
            )
            capture = CalvinSceneObserver(self.adapter).capture_scene()
            success = bool(result.success)
            verification = {
                "backend": "calvin_oracle",
                "status": "success" if success else "failure",
                "reason": (
                    "predicate_satisfied"
                    if success
                    else "predicate_not_satisfied"
                ),
            }
            recommended = (
                ("verify_goal", "inspect_scene")
                if success
                else (
                    "inspect_scene",
                    "ground_targets",
                    "approach_target",
                    "replan",
                )
            )
            return SkillExecutionResult(
                policy_id=decision.policy.policy_id,
                success=success,
                termination_reason=result.termination_reason,
                steps=result.steps_executed,
                verification=verification,
                scene_capture=capture,
                state_delta={
                    "gripper": capture.robot_summary.get("gripper", "unknown"),
                    "holding_object": capture.robot_summary.get(
                        "holding_object", "unknown"
                    ),
                    "target_moved": "unknown",
                },
                recommended_actions=recommended,
                frames=tuple(result.frames),
                video_fps=self.video_fps,
                metadata={
                    "policy_session_id": policy_session_id,
                    "environment_episode_id": self.environment_episode_id,
                    "episode_seed": seed,
                    "replan_interval": decision.skill.replan_interval,
                    "success_step": result.success_step,
                    "environment_done": result.environment_done,
                    "latencies_ms": result.latencies_ms,
                    "replan_steps": result.replan_steps,
                    "robot_recovery": recovery,
                },
            )
        except SkillExecutorPortError:
            raise
        except PolicyConnectionError as exc:
            raise SkillExecutorUnavailable(str(exc))
        except Exception as exc:
            raise SkillExecutorPortError(
                "CALVIN skill execution failed: {}: {}".format(
                    type(exc).__name__, exc
                )
            )

    def _restore_robot_for_retry(self, skill_id, attempt):
        if int(attempt) <= 1:
            return {"performed": False, "reason": "first_attempt"}
        if skill_id in STATEFUL_RETRY_SKILLS:
            return {
                "performed": False,
                "reason": "preserve_stateful_skill_robot_state",
            }
        if self.initial_robot_observation is None:
            return {"performed": False, "reason": "initial_state_unavailable"}

        before = np.asarray(
            self.adapter.current_observation["robot_obs"], dtype=np.float64
        )
        self.adapter.restore_robot(self.initial_robot_observation)
        after = np.asarray(
            self.adapter.current_observation["robot_obs"], dtype=np.float64
        )
        joint_slice = slice(7, 14)
        return {
            "performed": True,
            "reason": "retry_initial_joint_restore",
            "max_joint_delta_before_rad": float(
                np.max(
                    np.abs(
                        before[joint_slice]
                        - self.initial_robot_observation[joint_slice]
                    )
                )
            ),
            "max_joint_error_after_rad": float(
                np.max(
                    np.abs(
                        after[joint_slice]
                        - self.initial_robot_observation[joint_slice]
                    )
                )
            ),
            "gripper_command": int(np.sign(self.initial_robot_observation[14])),
            "scene_reset": False,
        }

    def _mask_roles(self, groundings):
        result = {}
        for grounding in groundings:
            if (
                grounding.source != "calvin_gt"
                or grounding.representation != GroundingRepresentation.SEMANTIC
            ):
                raise SkillExecutorPortError(
                    "first-version CALVIN execution requires calvin_gt semantic grounding"
                )
            try:
                selector = dict(self.vocabulary.selectors[grounding.semantic_id])
            except KeyError:
                raise SkillExecutorPortError(
                    "semantic_id {!r} is absent from execution vocabulary".format(
                        grounding.semantic_id
                    )
                )
            if "link" in selector:
                selector["links"] = [selector.pop("link")]
            result[grounding.role.value] = [selector]
        return result


class _PolicyResponseAdapter:
    def __init__(self, client):
        self.client = client

    def reset(self, **kwargs):
        return self.client.reset(**kwargs)

    def step(self, episode_id, observation, step):
        response = self.client.step(episode_id, observation, step)
        action = np.asarray(response["action"], dtype=np.float32).reshape(-1).copy()
        return action, dict(response.get("metadata", {}))
