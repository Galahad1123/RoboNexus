#!/usr/bin/env python3
"""Host one live CALVIN Agent session for a Qwen-Agent process."""

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
from backends.calvin.runtime import CalvinToolSessionServer  # noqa: E402
from backends.calvin.verification.goal_verifier import CalvinOracleGoalVerifier  # noqa: E402
from backends.local_artifact_store import LocalArtifactStore  # noqa: E402
from calvin.evaluation.remote_policy.calvin_env_adapter import CalvinEnvAdapter  # noqa: E402
from calvin.evaluation.remote_policy.task_suite import ConfiguredCalvinSuite  # noqa: E402
from core.domain import GoalInput, ObservationRef, RuntimeLimits  # noqa: E402
from core.logging import JsonlEventLogger  # noqa: E402
from core.policies import PolicyClientPool, PolicyRouter, load_policy_catalog  # noqa: E402
from core.runtime import AgentSession  # noqa: E402
from core.skills import SkillRegistry, load_skill_registry  # noqa: E402
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8110)
    parser.add_argument("--task", default="lift_red_block_table")
    parser.add_argument(
        "--skill",
        action="append",
        default=[],
        help="allowed goal skill; repeat for a multi-subtask goal (defaults to --task)",
    )
    parser.add_argument("--goal", help="goal instruction; defaults to the initial task instruction")
    parser.add_argument("--disable-approach", action="store_true")
    parser.add_argument("--environment-step-limit", type=int, default=1000)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--suite",
        type=Path,
        default=WORKSPACE / "datasets/calvin_d_10/configs/evaluation.yaml",
    )
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
        default=WORKSPACE / "outputs/calvin_agent/qwen_single_skill",
    )
    parser.add_argument("--show-gui", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = LocalArtifactStore(args.output_dir / "artifacts")
    events = JsonlEventLogger(args.output_dir / "events.jsonl", "qwen-single-skill")
    full_registry = load_skill_registry(args.skills_config)
    allowed_skill_ids = args.skill or [args.task]
    if len(set(allowed_skill_ids)) != len(allowed_skill_ids):
        parser.error("--skill values must be unique")
    selected_skills = {
        skill_id: full_registry.get(skill_id) for skill_id in allowed_skill_ids
    }
    registry = SkillRegistry(
        full_registry.name + "_single_skill",
        full_registry.version,
        full_registry.defaults,
        selected_skills,
        full_registry.source,
    )
    catalog = load_policy_catalog(
        args.policies_config, required_policy_ids=full_registry.policy_groups()
    )
    catalog.validate_runtime(WORKSPACE, full_registry.policy_groups())
    suite = ConfiguredCalvinSuite(args.suite)
    episode = suite.episodes(args.task)[args.episode_index]
    adapter = CalvinEnvAdapter.create(
        suite.dataset_split,
        CALVIN_METADATA,
        show_gui=args.show_gui,
        use_egl=not args.show_gui,
    )
    pool = PolicyClientPool(catalog, connect_timeout=300.0)
    server = None
    try:
        robot_obs, scene_obs = episode.load_initial_state()
        observation = adapter.reset(robot_obs, scene_obs)
        zero_action = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        for _ in range(episode.settle_steps):
            observation, _, _, _ = adapter.step(zero_action)
        observer = CalvinSceneObserver(adapter)
        capture = observer.capture_scene()
        goal_slug = "__".join(allowed_skill_ids)
        session_id = "qwen-{}-{:03d}".format(goal_slug, args.episode_index)
        static = artifacts.save_image(session_id, "initial_static", capture.static_rgb)
        wrist = artifacts.save_image(session_id, "initial_wrist", capture.wrist_rgb)
        initial = ObservationRef(
            "obs_000001", 0, capture.environment_steps, static.uri, wrist.uri
        )
        session = AgentSession(
            GoalInput(
                "goal-{}-{:03d}".format(goal_slug, args.episode_index),
                args.goal or episode.instruction,
                {
                    "mode": "skill_sequence",
                    "predicates": [
                        {"name": selected_skills[skill_id].oracle_task}
                        for skill_id in allowed_skill_ids
                    ]
                },
                initial,
                limits=RuntimeLimits(
                    tool_call_limit=max(32, 20 * len(allowed_skill_ids)),
                    environment_step_limit=args.environment_step_limit
                ),
                metadata={"allowed_skill_ids": allowed_skill_ids},
            ),
            session_id=session_id,
            events=events,
        )
        task_oracle = suite.create_task_oracle()
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
        tools = [
            InspectSceneTool(session, observer, artifacts),
            GroundTargetsTool(session, registry, grounder),
            ExecuteSkillTool(session, registry, executor, artifacts),
            VerifyGoalTool(session, CalvinOracleGoalVerifier(adapter, task_oracle)),
            AbortGoalTool(session, enforce_attempt_exhaustion=True),
        ]
        if not args.disable_approach:
            tools.insert(
                3,
                ApproachTargetTool(
                    session, CalvinApproachController(adapter), artifacts
                ),
            )
        server = CalvinToolSessionServer(
            args.host,
            args.port,
            session,
            tools,
            artifacts,
            {
                "goal_id": session.goal.goal_id,
                "goal_instruction": session.goal.instruction,
                "success_spec": session.goal.success_spec,
                "allowed_skill_ids": allowed_skill_ids,
                "required_grounding_roles": {
                    skill_id: list(selected_skills[skill_id].grounding_roles)
                    for skill_id in allowed_skill_ids
                },
                "approach_enabled": not args.disable_approach,
                "initial_observation": {
                    "observation_id": initial.observation_id,
                    "static_image": str(artifacts.resolve(static.uri)),
                    "wrist_image": str(artifacts.resolve(wrist.uri)),
                },
                "target_reference_image": None,
                "semantic_vocabulary": list(grounder.capabilities()["semantic_vocabulary"]),
                "output_dir": str(args.output_dir.resolve()),
            },
            on_terminal=pool.close_all,
        )
        print(
            json.dumps(
                {
                    "runtime_url": "http://{}:{}".format(args.host, args.port),
                    "session_id": session_id,
                    "goal": session.goal.instruction,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        server.serve_forever()
    finally:
        report = {
            "session_status": session.status.value if "session" in locals() else "not_started",
            "termination_reason": session.termination_reason if "session" in locals() else None,
            "environment_steps": adapter.environment_steps,
        }
        (args.output_dir / "runtime_result.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
        )
        if server is not None:
            server.close()
        pool.close_all()
        adapter.close()
        events.close()


if __name__ == "__main__":
    main()
