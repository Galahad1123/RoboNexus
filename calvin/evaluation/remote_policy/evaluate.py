"""Run a configuration-driven CALVIN suite against a remote policy server."""

import argparse
import json
import logging
from pathlib import Path

try:
    from .calvin_env_adapter import CalvinEnvAdapter
    from .model_client import ModelClient
    from .result_writer import ResultWriter
    from .rollout_engine import RolloutEngine, derive_episode_seed
    from .task_suite import ConfiguredCalvinSuite
    from .masking import CalvinGTMaskProvider, MaskOverlay, MaskPerturbation
except ImportError:
    # Retain direct-script execution for development inside this directory.
    from calvin_env_adapter import CalvinEnvAdapter
    from model_client import ModelClient
    from result_writer import ResultWriter
    from rollout_engine import RolloutEngine, derive_episode_seed
    from task_suite import ConfiguredCalvinSuite
    from masking import CalvinGTMaskProvider, MaskOverlay, MaskPerturbation


def run(args):
    suite = ConfiguredCalvinSuite(args.suite)
    configured_mask_style = suite.config.get("mask_style", {})
    mask_alpha = (
        args.mask_alpha
        if args.mask_alpha is not None
        else float(configured_mask_style.get("alpha", 0.45))
    )
    mask_outline_px = (
        args.mask_outline_px
        if args.mask_outline_px is not None
        else int(configured_mask_style.get("outline_width_px", 0))
    )
    mask_outline_color = (
        args.mask_outline_color
        if args.mask_outline_color is not None
        else configured_mask_style.get("outline_color", [0, 0, 0])
    )
    mask_outline_halo_px = (
        args.mask_outline_halo_px
        if args.mask_outline_halo_px is not None
        else int(configured_mask_style.get("outline_halo_width_px", 0))
    )
    mask_outline_halo_color = (
        args.mask_outline_halo_color
        if args.mask_outline_halo_color is not None
        else configured_mask_style.get("outline_halo_color", [255, 255, 255])
    )
    tasks = args.task or suite.task_names()
    unknown = sorted(set(tasks) - set(suite.task_names()))
    if unknown:
        raise ValueError("Tasks are not in suite: {}".format(unknown))
    unknown_variants = sorted(
        set(args.episode_variant or ()) - set(suite.episode_variant_names())
    )
    if unknown_variants:
        raise ValueError(
            "Episode variants are not in suite: {}".format(unknown_variants)
        )

    client = ModelClient(host=args.host, port=args.port)
    adapter = None
    event_logger = None
    try:
        adapter = CalvinEnvAdapter.create(
            suite.dataset_split,
            client.metadata,
            show_gui=args.show_gui,
            use_egl=not args.show_gui,
        )
        task_oracle = suite.create_task_oracle()
        mask_provider = None
        mask_overlay = None
        if args.mask_provider == "gt":
            mask_provider = CalvinGTMaskProvider(adapter.env)
            mask_overlay = MaskOverlay(
                alpha=mask_alpha,
                seed=args.seed,
                outline_width_px=mask_outline_px,
                outline_color=mask_outline_color,
                outline_colors=configured_mask_style.get("outline_colors"),
                outline_halo_width_px=mask_outline_halo_px,
                outline_halo_widths=configured_mask_style.get(
                    "outline_halo_widths"
                ),
                outline_halo_color=mask_outline_halo_color,
                perturbation=MaskPerturbation(
                    max_shift_px=args.mask_max_shift_px,
                    morphology_px=args.mask_morphology_px,
                    dropout_probability=args.mask_dropout_probability,
                ),
            )
        writer = ResultWriter(
            args.output_dir,
            {
                "suite": str(Path(args.suite).resolve()),
                "tasks": tasks,
                "episodes_per_task": args.episodes_per_task,
                "episode_variants": args.episode_variant,
                "seed": args.seed,
                "replan_interval": args.replan_interval,
                "max_attempts": args.max_attempts,
                "retry_recovery": args.retry_recovery,
                "server_metadata": client.metadata,
                "mask_provider": args.mask_provider,
                "mask_alpha": mask_alpha,
                "mask_outline_px": mask_outline_px,
                "mask_outline_color": mask_outline_color,
                "mask_outline_halo_px": mask_outline_halo_px,
                "mask_outline_halo_color": mask_outline_halo_color,
                "mask_max_shift_px": args.mask_max_shift_px,
                "mask_morphology_px": args.mask_morphology_px,
                "mask_dropout_probability": args.mask_dropout_probability,
            },
            overwrite=args.overwrite,
        )
        engine = RolloutEngine(
            adapter,
            client,
            task_oracle,
            replan_interval=args.replan_interval,
            base_seed=args.seed,
            video_dir=Path(args.output_dir) / "videos" if args.save_video else None,
            video_fps=suite.manifest["frequency_hz"],
            mask_provider=mask_provider,
            mask_overlay=mask_overlay,
        )
        runner = engine
        if args.max_attempts > 1:
            from backends.calvin.event_logger import JsonlEventLogger
            from backends.calvin.retry_runner import CalvinOracleRetryAgent

            event_logger = JsonlEventLogger(
                Path(args.output_dir) / "events.jsonl",
                Path(args.output_dir).name,
            )
            recovery = None
            if args.retry_recovery == "oracle-approach":
                from backends.calvin.approach_recovery import (
                    CalvinOracleApproachRecovery,
                )

                recovery = CalvinOracleApproachRecovery(adapter)
            runner = CalvinOracleRetryAgent(
                engine,
                args.max_attempts,
                recovery=recovery,
                event_logger=event_logger,
            )

        for task in tasks:
            selection_seed = derive_episode_seed(
                args.seed, "episode-selection:{}".format(task)
            ) % (2**32 - 1)
            episodes = suite.episodes(
                task,
                count=args.episodes_per_task,
                seed=selection_seed,
                episode_variants=args.episode_variant,
            )
            for episode in episodes:
                logging.info("Starting %s (%s)", episode.episode_id, task)
                try:
                    record = runner.run(episode)
                except Exception as exc:
                    logging.exception("Episode %s failed", episode.episode_id)
                    record = {
                        "episode_id": episode.episode_id,
                        "task": episode.task,
                        "episode_variant": episode.episode_variant,
                        "instruction": episode.instruction,
                        "success": False,
                        "termination_reason": "exception",
                        "error": "{}: {}".format(type(exc).__name__, exc),
                        "replan_interval": args.replan_interval,
                        "episode_seed": derive_episode_seed(
                            args.seed, episode.episode_id
                        ),
                    }
                    writer.append(record)
                    if args.fail_fast:
                        raise
                else:
                    writer.append(record)
                logging.info(
                    "Episode result: success=%s reason=%s",
                    record["success"],
                    record["termination_reason"],
                )
        summary = writer.write_summary()
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        if event_logger is not None:
            event_logger.close()
        if adapter is not None:
            adapter.close()
        client.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--task", action="append")
    parser.add_argument(
        "--episode-variant",
        action="append",
        help="repeat to evaluate only selected manifest episode variants",
    )
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--replan-interval", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing evaluation output directory",
    )
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--show-gui", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="1 uses the clean rollout; >1 enables agent-side oracle retry in place",
    )
    parser.add_argument(
        "--retry-recovery",
        choices=("none", "oracle-approach"),
        default="none",
        help="recovery before retry; oracle-approach localizes the configured target and moves above it",
    )
    parser.add_argument(
        "--mask-provider",
        choices=("none", "gt"),
        default="none",
        help="optional image overlay source; clean evaluation remains the default",
    )
    parser.add_argument("--mask-alpha", type=float)
    parser.add_argument(
        "--mask-outline-px",
        type=int,
        help="opaque role-color outline width; defaults to the suite mask style",
    )
    parser.add_argument("--mask-outline-color", type=int, nargs=3)
    parser.add_argument("--mask-outline-halo-px", type=int)
    parser.add_argument("--mask-outline-halo-color", type=int, nargs=3)
    parser.add_argument("--mask-max-shift-px", type=int, default=0)
    parser.add_argument(
        "--mask-morphology-px",
        type=int,
        default=0,
        help="positive dilates and negative erodes the GT mask",
    )
    parser.add_argument("--mask-dropout-probability", type=float, default=0.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be >= 1")
    if args.mask_alpha is not None and not 0.0 <= args.mask_alpha <= 1.0:
        parser.error("--mask-alpha must be in [0, 1]")
    if not 0.0 <= args.mask_dropout_probability <= 1.0:
        parser.error("--mask-dropout-probability must be in [0, 1]")
    if args.mask_outline_px is not None and args.mask_outline_px < 0:
        parser.error("--mask-outline-px must be >= 0")
    if args.mask_outline_halo_px is not None and args.mask_outline_halo_px < 0:
        parser.error("--mask-outline-halo-px must be >= 0")
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), force=True)
    return args


if __name__ == "__main__":
    run(parse_args())
