#!/usr/bin/env python3
"""Run one scripted multi-skill goal against four routed CALVIN DiT servers."""

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

from backends.calvin.execution.goal_attempt_runner import CalvinGoalAttemptRunner  # noqa: E402
from backends.calvin.verification.state_inspector import CalvinStateInspector  # noqa: E402
from calvin.evaluation.remote_policy.calvin_env_adapter import CalvinEnvAdapter  # noqa: E402
from calvin.evaluation.remote_policy.masking import (  # noqa: E402
    CalvinGTMaskProvider,
    MaskOverlay,
    MaskPerturbation,
)
from calvin.evaluation.remote_policy.task_suite import ConfiguredCalvinSuite  # noqa: E402
from core.logging import JsonlEventLogger  # noqa: E402
from core.legacy.orchestration import (  # noqa: E402
    GoalExecutionConfig,
    GoalOrchestrator,
)
from core.legacy.planning import GoalSpec, ScriptedPlanner  # noqa: E402
from core.policies import (  # noqa: E402
    PolicyClientPool,
    PolicyRouter,
    load_policy_catalog,
)
from core.skills import load_skill_registry  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-id", default="calvin-scripted-goal")
    parser.add_argument("--goal", required=True, help="human-readable goal text")
    parser.add_argument(
        "--skill", action="append", required=True, help="canonical skill; repeat in order"
    )
    parser.add_argument(
        "--replan-script",
        action="append",
        default=[],
        help="comma-separated canonical skills; repeat for up to two replans",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=WORKSPACE / "datasets/calvin_d_10/configs/evaluation.yaml",
        help="suite used only to select the initial CALVIN state",
    )
    parser.add_argument("--initial-task", default="open_drawer")
    parser.add_argument("--episode-index", type=int, default=0)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-replans", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--connect-timeout", type=float, default=300.0)
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="save one static-camera MP4 for every executed attempt",
    )
    parser.add_argument(
        "--mask-provider",
        choices=("none", "gt"),
        default="none",
        help="image overlay source; gt uses CALVIN simulator segmentation",
    )
    parser.add_argument("--mask-alpha", type=float)
    parser.add_argument("--mask-outline-px", type=int)
    parser.add_argument("--mask-outline-color", type=int, nargs=3)
    parser.add_argument("--mask-outline-halo-px", type=int)
    parser.add_argument("--mask-outline-halo-color", type=int, nargs=3)
    parser.add_argument("--mask-max-shift-px", type=int, default=0)
    parser.add_argument("--mask-morphology-px", type=int, default=0)
    parser.add_argument("--mask-dropout-probability", type=float, default=0.0)
    parser.add_argument("--show-gui", action="store_true")
    args = parser.parse_args()
    _validate_mask_args(parser, args)

    registry = load_skill_registry(args.skills_config)
    catalog = load_policy_catalog(
        args.policies_config, required_policy_ids=registry.policy_groups()
    )
    catalog.validate_runtime(WORKSPACE, registry.policy_groups())
    replan_scripts = [
        [skill.strip() for skill in script.split(",") if skill.strip()]
        for script in args.replan_script
    ]
    planner = ScriptedPlanner(args.skill, replan_scripts, registry)
    router = PolicyRouter(registry, catalog)
    suite = ConfiguredCalvinSuite(args.suite)
    episodes = suite.episodes(args.initial_task)
    if not 0 <= args.episode_index < len(episodes):
        raise IndexError(
            "episode-index {} outside [0, {})".format(
                args.episode_index, len(episodes)
            )
        )
    initial_episode = episodes[args.episode_index]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = JsonlEventLogger(args.output_dir / "events.jsonl", args.goal_id)
    pool = PolicyClientPool(catalog, connect_timeout=args.connect_timeout)
    adapter = None
    try:
        clients = pool.connect_all(registry.policy_groups())
        pool.health_check_all(registry.policy_groups())
        metadata = clients["general"].metadata
        adapter = CalvinEnvAdapter.create(
            suite.dataset_split,
            metadata,
            show_gui=args.show_gui,
            use_egl=not args.show_gui,
        )
        robot_obs, scene_obs = initial_episode.load_initial_state()
        observation = adapter.reset(robot_obs, scene_obs)
        zero_action = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        for _ in range(initial_episode.settle_steps):
            observation, _, _, _ = adapter.step(zero_action)

        mask_provider, mask_overlay, mask_config = _create_masking(
            args, suite, adapter
        )

        attempt_runner = CalvinGoalAttemptRunner(
            adapter=adapter,
            task_oracle=suite.create_task_oracle(),
            skill_registry=registry,
            policy_router=router,
            client_pool=pool,
            event_logger=events,
            mask_provider=mask_provider,
            mask_overlay=mask_overlay,
            mask_roles_by_skill={initial_episode.task: initial_episode.mask_roles},
            video_dir=args.output_dir / "videos" if args.save_video else None,
            video_fps=suite.manifest.get("frequency_hz", 30),
        )
        orchestrator = GoalOrchestrator(
            planner=planner,
            attempt_runner=attempt_runner,
            config=GoalExecutionConfig(
                max_attempts_per_skill=args.max_attempts,
                max_replans=args.max_replans,
                base_seed=args.seed,
            ),
            events=events,
            state_inspector=CalvinStateInspector(),
        )
        result = orchestrator.run(
            GoalSpec(args.goal_id, args.goal), initial_observation=observation
        )
        report = _report(result, initial_episode, mask_config)
        (args.output_dir / "result.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str)
        )
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    finally:
        if adapter is not None:
            adapter.close()
        pool.close_all()
        events.close()


def _report(result, initial_episode, mask_config):
    return {
        "goal_id": result.goal.goal_id,
        "goal": result.goal.instruction,
        "status": result.status.value,
        "success": result.success,
        "termination_reason": result.termination_reason,
        "completed_skill_ids": list(result.completed_skill_ids),
        "attempts_executed": result.attempts_executed,
        "replans_used": result.replans_used,
        "initial_episode": initial_episode.episode_id,
        "initial_task": initial_episode.task,
        "mask": mask_config,
        "skill_records": [
            {
                "skill_id": record.skill_id,
                "plan_step": record.plan_step,
                "replans_used": record.replans_used,
                "skipped": record.skipped,
                "inspection_reason": record.inspection_reason,
                "attempts": [
                    {
                        "success": attempt.success,
                        "steps": attempt.steps,
                        "termination_reason": attempt.termination_reason,
                        "metadata": attempt.metadata,
                    }
                    for attempt in record.attempts
                ],
            }
            for record in result.skill_records
        ],
    }


def _validate_mask_args(parser, args):
    if args.mask_alpha is not None and not 0.0 <= args.mask_alpha <= 1.0:
        parser.error("--mask-alpha must be in [0, 1]")
    if not 0.0 <= args.mask_dropout_probability <= 1.0:
        parser.error("--mask-dropout-probability must be in [0, 1]")
    if args.mask_outline_px is not None and args.mask_outline_px < 0:
        parser.error("--mask-outline-px must be >= 0")
    if args.mask_outline_halo_px is not None and args.mask_outline_halo_px < 0:
        parser.error("--mask-outline-halo-px must be >= 0")


def _create_masking(args, suite, adapter):
    style = suite.config.get("mask_style", {})
    config = {
        "provider": args.mask_provider,
        "alpha": (
            args.mask_alpha
            if args.mask_alpha is not None
            else float(style.get("alpha", 0.45))
        ),
        "outline_width_px": (
            args.mask_outline_px
            if args.mask_outline_px is not None
            else int(style.get("outline_width_px", 0))
        ),
        "outline_color": (
            args.mask_outline_color
            if args.mask_outline_color is not None
            else style.get("outline_color", [0, 0, 0])
        ),
        "outline_halo_width_px": (
            args.mask_outline_halo_px
            if args.mask_outline_halo_px is not None
            else int(style.get("outline_halo_width_px", 0))
        ),
        "outline_halo_color": (
            args.mask_outline_halo_color
            if args.mask_outline_halo_color is not None
            else style.get("outline_halo_color", [255, 255, 255])
        ),
        "max_shift_px": args.mask_max_shift_px,
        "morphology_px": args.mask_morphology_px,
        "dropout_probability": args.mask_dropout_probability,
    }
    if args.mask_provider == "none":
        return None, None, config
    provider = CalvinGTMaskProvider(adapter.env)
    overlay = MaskOverlay(
        alpha=config["alpha"],
        seed=args.seed,
        outline_width_px=config["outline_width_px"],
        outline_color=config["outline_color"],
        outline_colors=style.get("outline_colors"),
        outline_halo_width_px=config["outline_halo_width_px"],
        outline_halo_widths=style.get("outline_halo_widths"),
        outline_halo_color=config["outline_halo_color"],
        perturbation=MaskPerturbation(
            max_shift_px=config["max_shift_px"],
            morphology_px=config["morphology_px"],
            dropout_probability=config["dropout_probability"],
        ),
    )
    return provider, overlay, config


if __name__ == "__main__":
    main()
