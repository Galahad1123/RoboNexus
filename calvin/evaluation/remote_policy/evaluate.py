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
except ImportError:
    # Retain direct-script execution for development inside this directory.
    from calvin_env_adapter import CalvinEnvAdapter
    from model_client import ModelClient
    from result_writer import ResultWriter
    from rollout_engine import RolloutEngine, derive_episode_seed
    from task_suite import ConfiguredCalvinSuite


def run(args):
    suite = ConfiguredCalvinSuite(args.suite)
    tasks = args.task or suite.task_names()
    unknown = sorted(set(tasks) - set(suite.task_names()))
    if unknown:
        raise ValueError("Tasks are not in suite: {}".format(unknown))

    client = ModelClient(host=args.host, port=args.port)
    adapter = None
    try:
        adapter = CalvinEnvAdapter.create(
            suite.dataset_split,
            client.metadata,
            show_gui=args.show_gui,
            use_egl=not args.show_gui,
        )
        task_oracle = suite.create_task_oracle()
        writer = ResultWriter(
            args.output_dir,
            {
                "suite": str(Path(args.suite).resolve()),
                "tasks": tasks,
                "episodes_per_task": args.episodes_per_task,
                "seed": args.seed,
                "replan_interval": args.replan_interval,
                "server_metadata": client.metadata,
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
        )

        for task in tasks:
            selection_seed = derive_episode_seed(
                args.seed, "episode-selection:{}".format(task)
            ) % (2**32 - 1)
            episodes = suite.episodes(
                task,
                count=args.episodes_per_task,
                seed=selection_seed,
            )
            for episode in episodes:
                logging.info("Starting %s (%s)", episode.episode_id, task)
                try:
                    record = engine.run(episode)
                except Exception as exc:
                    logging.exception("Episode %s failed", episode.episode_id)
                    record = {
                        "episode_id": episode.episode_id,
                        "task": episode.task,
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
        if adapter is not None:
            adapter.close()
        client.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--task", action="append")
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
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), force=True)
    return args


if __name__ == "__main__":
    run(parse_args())
