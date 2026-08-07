#!/usr/bin/env python3
"""Exercise the six Agent tools against a real CALVIN-D environment."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[1]
AGENT_ROOT = WORKSPACE / "RoboNexus-Agent"
for path in (AGENT_ROOT, WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backends.calvin.environment.scene_observer import CalvinSceneObserver  # noqa: E402
from backends.calvin.execution.skill_executor import CalvinSkillExecutor  # noqa: E402
from backends.calvin.grounding.oracle_grounder import CalvinOracleGrounder  # noqa: E402
from backends.calvin.recovery.approach_controller import CalvinApproachController  # noqa: E402
from backends.calvin.verification.goal_verifier import CalvinOracleGoalVerifier  # noqa: E402
from backends.local_artifact_store import LocalArtifactStore  # noqa: E402
from calvin.evaluation.remote_policy.calvin_env_adapter import CalvinEnvAdapter  # noqa: E402
from calvin.evaluation.remote_policy.task_suite import ConfiguredCalvinSuite  # noqa: E402
from core.domain import GoalInput, ObservationRef, RuntimeLimits  # noqa: E402
from core.logging import JsonlEventLogger  # noqa: E402
from core.policies import PolicyClientPool, PolicyRouter, load_policy_catalog  # noqa: E402
from core.runtime import AgentSession  # noqa: E402
from core.skills import load_skill_registry  # noqa: E402
from core.tools import (  # noqa: E402
    AbortGoalTool,
    ApproachTargetTool,
    ExecuteSkillTool,
    GroundTargetsTool,
    InspectSceneTool,
    VerifyGoalTool,
)


CALVIN_METADATA = {
    "image_keys": ["observation.images.static", "observation.images.gripper"],
    "image_shapes": {
        "observation.images.static": [3, 200, 200],
        "observation.images.gripper": [3, 200, 200],
    },
    "state_dim": 15,
    "action_dim": 7,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=WORKSPACE / "datasets/calvin_d_10/configs/evaluation.yaml",
    )
    parser.add_argument("--task", default="lift_red_block_table")
    parser.add_argument("--semantic-id", default="block_red")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--xy-threshold-m", type=float, default=0.10)
    parser.add_argument("--execute-steps", type=int, default=120)
    parser.add_argument("--skip-execute", action="store_true")
    parser.add_argument(
        "--skills-config",
        type=Path,
        default=AGENT_ROOT / "configs/calvin/skills_d34.yaml",
    )
    parser.add_argument(
        "--policies-config",
        type=Path,
        default=AGENT_ROOT / "configs/calvin/policies.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "outputs/calvin_agent/tool_smoke",
    )
    parser.add_argument("--show-gui", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = LocalArtifactStore(args.output_dir / "artifacts")
    events = JsonlEventLogger(args.output_dir / "events.jsonl", "calvin-tool-smoke")
    suite = ConfiguredCalvinSuite(args.suite)
    episode = suite.episodes(args.task)[args.episode_index]
    registry = load_skill_registry(args.skills_config)
    adapter = CalvinEnvAdapter.create(
        suite.dataset_split,
        CALVIN_METADATA,
        show_gui=args.show_gui,
        use_egl=not args.show_gui,
    )
    task_oracle = suite.create_task_oracle()
    report = {
        "task": args.task,
        "episode_id": episode.episode_id,
        "tools": {},
        "execute_skipped": args.skip_execute,
    }
    pool = None
    try:
        _reset(adapter, episode)
        session, observer, verifier = _new_session(
            adapter, task_oracle, artifacts, events, args.task, "motion", 240
        )
        inspect_result = InspectSceneTool(session, observer, artifacts).call(
            {"reason": "real CALVIN tool smoke test"}
        )
        _record(report, "inspect_scene", inspect_result)
        _require_success("inspect_scene", inspect_result)

        grounder = CalvinOracleGrounder(adapter, artifacts, session.session_id)
        ground_result = GroundTargetsTool(session, registry, grounder).call(
            _ground_params(args.task, session.latest_observation.observation_id, args.semantic_id)
        )
        _record(report, "ground_targets", ground_result)
        _require_success("ground_targets", ground_result)
        grounding_id = ground_result.data["groundings"][0]["grounding_id"]

        approach_result = ApproachTargetTool(
            session, CalvinApproachController(adapter), artifacts
        ).call(
            {
                "grounding_id": grounding_id,
                "xy_threshold_m": args.xy_threshold_m,
                "intent": "approach the selected object before a retry",
            }
        )
        _record(report, "approach_target", approach_result)

        verify_result = VerifyGoalTool(session, verifier).call(
            {
                "claim": "the requested manipulation is complete",
                "observation_id": session.latest_observation.observation_id,
            }
        )
        _record(report, "verify_goal_before_execute", verify_result)
        if session.status.value == "active":
            abort_result = AbortGoalTool(session).call(
                {
                    "reason_code": "repeated_execution_failure",
                    "summary": "motion-only smoke scene intentionally ended before DiT execution",
                    "last_observation_id": session.latest_observation.observation_id,
                }
            )
            _record(report, "abort_goal", abort_result)

        if not args.skip_execute:
            _reset(adapter, episode)
            execute_session, _, execute_verifier = _new_session(
                adapter,
                task_oracle,
                artifacts,
                events,
                args.task,
                "execute",
                args.execute_steps + 20,
            )
            execute_grounder = CalvinOracleGrounder(
                adapter, artifacts, execute_session.session_id
            )
            execute_ground = GroundTargetsTool(
                execute_session, registry, execute_grounder
            ).call(
                _ground_params(
                    args.task,
                    execute_session.latest_observation.observation_id,
                    args.semantic_id,
                )
            )
            _record(report, "execute_ground_targets", execute_ground)
            _require_success("execute_ground_targets", execute_ground)
            execute_grounding_id = execute_ground.data["groundings"][0]["grounding_id"]

            catalog = load_policy_catalog(
                args.policies_config, required_policy_ids=registry.policy_groups()
            )
            catalog.validate_runtime(WORKSPACE, registry.policy_groups())
            router = PolicyRouter(registry, catalog)
            pool = PolicyClientPool(catalog, connect_timeout=300.0)
            executor = CalvinSkillExecutor(
                adapter,
                task_oracle,
                router,
                pool,
                event_logger=events,
                video_fps=suite.manifest.get("frequency_hz", 30),
            )
            execute_result = ExecuteSkillTool(
                execute_session, registry, executor, artifacts
            ).call(
                {
                    "skill_id": args.task,
                    "grounding_ids": [execute_grounding_id],
                    "intent": "perform one real routed DiT attempt",
                }
            )
            _record(report, "execute_skill", execute_result)
            final_verify = VerifyGoalTool(execute_session, execute_verifier).call(
                {
                    "claim": "the requested manipulation is complete",
                    "observation_id": execute_session.latest_observation.observation_id,
                }
            )
            _record(report, "verify_goal_after_execute", final_verify)
            report["execute_policy_success"] = bool(
                execute_result.data.get("success", False)
            )

        report["smoke_completed"] = True
    except Exception as exc:
        report["smoke_completed"] = False
        report["fatal_error"] = "{}: {}".format(type(exc).__name__, exc)
        raise
    finally:
        if pool is not None:
            pool.close_all()
        adapter.close()
        events.close()
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
        )
        print(json.dumps(report, indent=2, sort_keys=True, default=str))


def _reset(adapter, episode):
    robot_obs, scene_obs = episode.load_initial_state()
    observation = adapter.reset(robot_obs, scene_obs)
    zero_action = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    for _ in range(episode.settle_steps):
        observation, _, _, _ = adapter.step(zero_action)
    return observation


def _new_session(adapter, task_oracle, artifacts, events, skill_id, suffix, step_limit):
    observer = CalvinSceneObserver(adapter)
    capture = observer.capture_scene()
    session_id = "calvin-tool-smoke-{}".format(suffix)
    static = artifacts.save_image(session_id, "initial_static", capture.static_rgb)
    wrist = artifacts.save_image(session_id, "initial_wrist", capture.wrist_rgb)
    initial = ObservationRef(
        "obs_000001",
        0,
        capture.environment_steps,
        static.uri,
        wrist.uri,
    )
    goal = GoalInput(
        "goal-{}".format(suffix),
        "complete CALVIN skill {}".format(skill_id),
        {"predicates": [{"name": skill_id}]},
        initial,
        limits=RuntimeLimits(environment_step_limit=int(step_limit)),
    )
    session = AgentSession(goal, session_id=session_id, events=events)
    verifier = CalvinOracleGoalVerifier(adapter, task_oracle)
    return session, observer, verifier


def _ground_params(skill_id, observation_id, semantic_id):
    return {
        "skill_id": skill_id,
        "observation_id": observation_id,
        "proposals": [
            {
                "role": "source",
                "representation": "semantic",
                "semantic_id": semantic_id,
                "description": "the planner-selected red block",
            }
        ],
    }


def _record(report, name, result):
    report["tools"][name] = result.to_dict()


def _require_success(name, result):
    if result.status.value != "succeeded":
        raise RuntimeError("{} did not succeed: {}".format(name, result.to_dict()))


if __name__ == "__main__":
    main()
