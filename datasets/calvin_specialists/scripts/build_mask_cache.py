#!/usr/bin/env python3
"""Render separate GT outline caches for audited specialist manifests."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = WORKSPACE / "datasets/calvin_specialists/configs/datasets.json"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from calvin.evaluation.remote_policy.calvin_env_adapter import CalvinEnvAdapter  # noqa: E402
from calvin.evaluation.remote_policy.masking import CalvinGTMaskProvider, MaskOverlay  # noqa: E402


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else WORKSPACE / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("training", "validation"), default="training")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task", action="append")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-episodes-per-task", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    if args.dataset not in config["datasets"]:
        raise ValueError("Unknown specialist dataset: {}".format(args.dataset))
    dataset_config = config["datasets"][args.dataset]
    manifest_path = WORKSPACE / "datasets" / args.dataset / "manifests/segments.json"
    manifest = json.loads(manifest_path.read_text())
    source_root = resolve(manifest["source_dataset"])
    source_split = source_root / args.split
    segments = list(manifest["splits"][args.split]["segments"])
    if args.task:
        segments = [row for row in segments if row["calvin_task"] in args.task]
    if args.max_episodes_per_task is not None:
        counts, selected = {}, []
        for row in segments:
            count = counts.get(row["calvin_task"], 0)
            if count < args.max_episodes_per_task:
                selected.append(row)
                counts[row["calvin_task"]] = count + 1
        segments = selected
    if args.max_episodes is not None:
        segments = segments[: args.max_episodes]
    if not segments:
        raise ValueError("No segments selected")

    output = (
        args.output
        or WORKSPACE / "datasets" / args.dataset / "mask_cache" / args.split
    ).resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError("{} exists; pass --overwrite".format(output))
        shutil.rmtree(str(output))
    output.mkdir(parents=True)

    metadata = {
        "image_keys": ["observation.images.static", "observation.images.gripper"],
        "image_shapes": {
            "observation.images.static": [3, 200, 200],
            "observation.images.gripper": [3, 200, 200],
        },
        "state_dim": 15,
        "action_dim": 7,
    }
    adapter = CalvinEnvAdapter.create(source_split, metadata, show_gui=False, use_egl=True)
    provider = CalvinGTMaskProvider(adapter.env)
    style = dataset_config["mask_style"]
    overlay = MaskOverlay(
        alpha=style["alpha"],
        outline_width_px=style["outline_width_px"],
        outline_color=style["outline_color"],
        outline_colors=style.get("outline_colors"),
        outline_halo_width_px=style.get("outline_halo_width_px", 0),
        outline_halo_color=style.get("outline_halo_color", [255, 255, 255]),
    )
    written = 0
    try:
        for segment in segments:
            episode = SimpleNamespace(
                episode_id=segment["segment_id"],
                task=segment["calvin_task"],
                mask_roles=segment["mask_roles"],
            )
            provider.reset(episode)
            episode_dir = output / segment["segment_id"]
            episode_dir.mkdir(parents=True)
            for frame_offset, source_index in enumerate(
                range(segment["start"], segment["end"] + 1)
            ):
                source_path = source_split / "episode_{:07d}.npz".format(source_index)
                with np.load(source_path) as source:
                    rgb_static = source["rgb_static"].copy()
                    rgb_gripper = source["rgb_gripper"].copy()
                    observation = adapter.reset(source["robot_obs"], source["scene_obs"])
                masks = provider.masks(observation)
                np.savez_compressed(
                    episode_dir / "episode_{:07d}.npz".format(source_index),
                    rgb_static=overlay.apply(
                        rgb_static,
                        masks.get("observation.images.static", {}),
                        frame_offset,
                    ),
                    rgb_gripper=overlay.apply(
                        rgb_gripper,
                        masks.get("observation.images.gripper", {}),
                        frame_offset,
                    ),
                )
                written += 1
    finally:
        adapter.close()

    info = {
        "dataset": args.dataset,
        "split": args.split,
        "manifest": str(manifest_path),
        "mask_style": style,
        "episodes": len(segments),
        "frames_written": written,
    }
    (output / "mask_cache_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
