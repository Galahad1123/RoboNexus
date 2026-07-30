#!/usr/bin/env python3
"""Build a separate GT-mask-overlay image cache from restored CALVIN states.

Run this with the CALVIN environment, not the MultiTask-DiT environment.  The
clean source NPZ files are read-only and never modified.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[2]
DEFAULT_SUITE = WORKSPACE / "datasets/calvin_d_10/configs/evaluation.yaml"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from calvin.evaluation.remote_policy.calvin_env_adapter import CalvinEnvAdapter  # noqa: E402
from calvin.evaluation.remote_policy.masking import (  # noqa: E402
    CalvinGTMaskProvider,
    MaskOverlay,
    MaskPerturbation,
)
from calvin.evaluation.remote_policy.task_suite import ConfiguredCalvinSuite  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--split", choices=("training", "validation"), default="training")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task", action="append")
    parser.add_argument("--max-episodes-per-task", type=int)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument(
        "--outline-px",
        type=int,
        default=0,
        help="opaque role-color outline width; 0 keeps fill-only overlays",
    )
    parser.add_argument("--max-shift-px", type=int, default=0)
    parser.add_argument("--morphology-px", type=int, default=0)
    parser.add_argument("--dropout-probability", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    suite = ConfiguredCalvinSuite(args.suite)
    suite.split = args.split
    suite.dataset_split = suite.source_dataset / args.split
    output = (
        args.output
        or WORKSPACE / "datasets/calvin_d_10/mask_cache" / args.split
    ).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError("{} is not empty; pass --overwrite".format(output))
    output.mkdir(parents=True, exist_ok=True)

    metadata = {
        "image_keys": [
            "observation.images.static",
            "observation.images.gripper",
        ],
        "image_shapes": {
            "observation.images.static": [3, 200, 200],
            "observation.images.gripper": [3, 200, 200],
        },
        "state_dim": 15,
        "action_dim": 7,
    }
    adapter = CalvinEnvAdapter.create(
        suite.dataset_split, metadata, show_gui=False, use_egl=True
    )
    provider = CalvinGTMaskProvider(adapter.env)
    overlay = MaskOverlay(
        alpha=args.alpha,
        seed=args.seed,
        outline_width_px=args.outline_px,
        perturbation=MaskPerturbation(
            max_shift_px=args.max_shift_px,
            morphology_px=args.morphology_px,
            dropout_probability=args.dropout_probability,
        ),
    )
    tasks = args.task or suite.task_names()
    written = 0
    try:
        for task in tasks:
            episodes = suite.episodes(task)
            if args.max_episodes_per_task is not None:
                episodes = episodes[: args.max_episodes_per_task]
            for episode in episodes:
                segment = next(
                    row
                    for row in suite.manifest["splits"][args.split]["segments"]
                    if row["segment_id"] == episode.episode_id
                )
                episode_dir = output / episode.episode_id
                episode_dir.mkdir(parents=True, exist_ok=True)
                provider.reset(episode)
                for frame_index, source_index in enumerate(
                    range(segment["start"], segment["end"] + 1)
                ):
                    destination = episode_dir / "episode_{:07d}.npz".format(
                        source_index
                    )
                    if destination.exists() and not args.overwrite:
                        continue
                    with np.load(
                        suite.dataset_split
                        / "episode_{:07d}.npz".format(source_index)
                    ) as source:
                        rgb_static = source["rgb_static"].copy()
                        rgb_gripper = source["rgb_gripper"].copy()
                        observation = adapter.reset(
                            source["robot_obs"], source["scene_obs"]
                        )
                    masks = provider.masks(observation)
                    np.savez_compressed(
                        destination,
                        rgb_static=overlay.apply(
                            rgb_static,
                            masks.get("observation.images.static", {}),
                            frame_index,
                        ),
                        rgb_gripper=overlay.apply(
                            rgb_gripper,
                            masks.get("observation.images.gripper", {}),
                            frame_index,
                        ),
                    )
                    written += 1
    finally:
        adapter.close()

    metadata_path = output / "mask_cache_info.json"
    metadata_path.write_text(
        json.dumps(
            {
                "split": args.split,
                "suite": str(args.suite.resolve()),
                "alpha": args.alpha,
                "outline_px": args.outline_px,
                "max_shift_px": args.max_shift_px,
                "morphology_px": args.morphology_px,
                "dropout_probability": args.dropout_probability,
                "seed": args.seed,
                "frames_written": written,
            },
            indent=2,
        )
        + "\n"
    )
    print(metadata_path)


if __name__ == "__main__":
    main()
