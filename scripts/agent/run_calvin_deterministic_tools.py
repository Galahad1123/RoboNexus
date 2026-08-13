#!/usr/bin/env python3
"""Run a no-Qwen deterministic Agent-tool loop in a real CALVIN scene."""

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
from core.runtime import (  # noqa: E402
    AgentSession,
    DeterministicLoopConfig,
    DeterministicToolLoop,
)
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
    parser.add_argument("--role", choices=("source", "target"), default="source")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--xy-threshold-m", type=float, default=0.10)
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
        default=WORKSPACE / "outputs/calvin_agent/deterministic_tool_loop",
    )
    parser.add_argument("--show-gui", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = LocalArtifactStore(args.output_dir / "artifacts")
    events = JsonlEventLogger(args.output_dir / "events.jsonl", "deterministic-tool-loop")
    registry = load_skill_registry(args.skills_config)
    catalog = load_policy_catalog(
        args.policies_config, required_policy_ids=registry.policy_groups()
    )
    catalog.validate_runtime(WORKSPACE, registry.policy_groups())
    suite = ConfiguredCalvinSuite(args.suite)
    episode = suite.episodes(args.task)[args.episode_index]
    adapter = CalvinEnvAdapter.create(
        suite.dataset_split,
        CALVIN_METADATA,
        show_gui=args.show_gui,
        use_egl=not args.show_gui,
    )
    pool = PolicyClientPool(catalog, connect_timeout=300.0)
    try:
        robot_obs, scene_obs = episode.load_initial_state()
        observation = adapter.reset(robot_obs, scene_obs)
        zero_action = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        for _ in range(episode.settle_steps):
            observation, _, _, _ = adapter.step(zero_action)

        observer = CalvinSceneObserver(adapter)
        capture = observer.capture_scene()
        session_id = "deterministic-{}-{:03d}".format(args.task, args.episode_index)
        static = artifacts.save_image(session_id, "initial_static", capture.static_rgb)
        wrist = artifacts.save_image(session_id, "initial_wrist", capture.wrist_rgb)
        initial = ObservationRef(
            "obs_000001", 0, capture.environment_steps, static.uri, wrist.uri
        )
        goal = GoalInput(
            "goal-{}-{:03d}".format(args.task, args.episode_index),
            episode.instruction,
            {"predicates": [{"name": episode.oracle_task}]},
            initial,
            limits=RuntimeLimits(
                skill_attempt_limit=args.max_attempts,
                tool_call_limit=32,
                environment_step_limit=max(500, episode.max_steps * args.max_attempts + 100),
            ),
        )
        session = AgentSession(goal, session_id=session_id, events=events)
        task_oracle = suite.create_task_oracle()
        verifier = CalvinOracleGoalVerifier(adapter, task_oracle)
        grounder = CalvinOracleGrounder(adapter, artifacts, session_id)
        executor = CalvinSkillExecutor(
            adapter,
            task_oracle,
            PolicyRouter(registry, catalog),
            pool,
            event_logger=events,
            video_fps=suite.manifest.get("frequency_hz", 30),
            environment_episode_id=episode.episode_id,
            initial_robot_observation=robot_obs,
        )
        loop = DeterministicToolLoop(
            session,
            InspectSceneTool(session, observer, artifacts),
            GroundTargetsTool(session, registry, grounder),
            ExecuteSkillTool(session, registry, executor, artifacts),
            ApproachTargetTool(session, CalvinApproachController(adapter), artifacts),
            VerifyGoalTool(session, verifier),
            AbortGoalTool(session),
        )
        result = loop.run(
            DeterministicLoopConfig(
                skill_id=args.task,
                proposals=(
                    {
                        "role": args.role,
                        "representation": "semantic",
                        "semantic_id": args.semantic_id,
                        "description": "deterministically selected {}".format(
                            args.semantic_id
                        ),
                    },
                ),
                max_attempts=args.max_attempts,
                approach_role=args.role,
                xy_threshold_m=args.xy_threshold_m,
            )
        )
        report = {
            "task": args.task,
            "episode_id": episode.episode_id,
            "episode_index": args.episode_index,
            "semantic_id": args.semantic_id,
            **result,
        }
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
        )
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0 if session.status.value == "succeeded" else 2
    finally:
        pool.close_all()
        adapter.close()
        events.close()


if __name__ == "__main__":
    sys.exit(main())
