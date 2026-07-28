#!/usr/bin/env python3
"""Convert a filtered CALVIN-D-10 manifest to LeRobotDataset v3.

Run with the ``multitask_dit_policy`` Python environment (LeRobot >= 0.6).
The same manifest and feature schema are used for video and image storage.
"""

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List

import numpy as np
from PIL import Image
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_DIR / "manifests" / "segments.json"

ROBOT_STATE_NAMES = [
    "tcp_pos_x",
    "tcp_pos_y",
    "tcp_pos_z",
    "tcp_orn_x",
    "tcp_orn_y",
    "tcp_orn_z",
    "gripper_opening_width",
    "robot_joint_0",
    "robot_joint_1",
    "robot_joint_2",
    "robot_joint_3",
    "robot_joint_4",
    "robot_joint_5",
    "robot_joint_6",
    "gripper_action",
]
ACTION_NAMES = [
    "delta_tcp_x",
    "delta_tcp_y",
    "delta_tcp_z",
    "delta_tcp_roll",
    "delta_tcp_pitch",
    "delta_tcp_yaw",
    "gripper_command",
]


def features(storage: str) -> Dict[str, dict]:
    visual_dtype = "video" if storage == "video" else "image"
    return {
        "observation.images.static": {
            "dtype": visual_dtype,
            "shape": [200, 200, 3],
            "names": ["height", "width", "channels"],
        },
        "observation.images.gripper": {
            "dtype": visual_dtype,
            "shape": [200, 200, 3],
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [15],
            "names": ROBOT_STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": ACTION_NAMES,
        },
        "action_is_valid": {
            "dtype": "bool",
            "shape": [1],
            "names": ["valid"],
        },
        "calvin.scene_obs": {
            "dtype": "float32",
            "shape": [24],
            "names": [f"scene_{index}" for index in range(24)],
        },
        "calvin.task_id": {
            "dtype": "int64",
            "shape": [1],
            "names": ["task_id"],
        },
        "calvin.source_frame_index": {
            "dtype": "int64",
            "shape": [1],
            "names": ["source_frame_index"],
        },
    }


def selected_segments(
    manifest: Dict[str, Any],
    split: str,
    requested_tasks: List[str] | None,
    max_episodes: int | None,
    max_episodes_per_task: int | None,
) -> List[dict]:
    segments = list(manifest["splits"][split]["segments"])
    if requested_tasks:
        unknown = sorted(set(requested_tasks) - {row["calvin_task"] for row in segments})
        if unknown:
            raise ValueError(f"tasks have no retained segments in {split}: {unknown}")
        segments = [row for row in segments if row["calvin_task"] in requested_tasks]
    if max_episodes_per_task is not None:
        task_counts: Dict[str, int] = {}
        balanced_segments = []
        for segment in segments:
            task = segment["calvin_task"]
            count = task_counts.get(task, 0)
            if count < max_episodes_per_task:
                balanced_segments.append(segment)
                task_counts[task] = count + 1
        segments = balanced_segments
    if max_episodes is not None:
        segments = segments[:max_episodes]
    return segments


def iter_episode_frames(source_split: Path, segment: dict) -> Iterable[dict]:
    for source_index in range(segment["start"], segment["end"] + 1):
        path = source_split / f"episode_{source_index:07d}.npz"
        with np.load(path) as source:
            # MultiTask DiT stacks camera features before model-side resize/crop,
            # so every camera must have the same HxW. Preserve the 200x200 static
            # image and upsample CALVIN's native 84x84 gripper image.
            gripper = np.asarray(
                Image.fromarray(source["rgb_gripper"]).resize((200, 200), resample=Image.Resampling.BILINEAR)
            )
            yield {
                "observation.images.static": source["rgb_static"],
                "observation.images.gripper": gripper,
                "observation.state": source["robot_obs"].astype(np.float32),
                "action": source["rel_actions"].astype(np.float32),
                "action_is_valid": np.asarray([source_index < segment["end"]], dtype=np.bool_),
                "calvin.scene_obs": source["scene_obs"].astype(np.float32),
                "calvin.task_id": np.asarray([segment["task_id"]], dtype=np.int64),
                "calvin.source_frame_index": np.asarray([source_index], dtype=np.int64),
                "task": segment["instruction"],
            }


def validate_source_frame(source_split: Path, segment: dict) -> None:
    first = source_split / f"episode_{segment['start']:07d}.npz"
    last = source_split / f"episode_{segment['end']:07d}.npz"
    if not first.is_file() or not last.is_file():
        raise FileNotFoundError(f"missing source frames for {segment['segment_id']}: {first}, {last}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=("training", "validation"), default="training")
    parser.add_argument("--storage", choices=("video", "image"), default="image")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task", action="append", help="repeat to convert only selected canonical tasks")
    parser.add_argument("--max-episodes", type=int, help="deterministic prefix for smoke tests")
    parser.add_argument("--max-episodes-per-task", type=int, help="retain at most N episodes for every task")
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    source_root = Path(manifest["source_dataset"])
    if not source_root.is_absolute():
        source_root = WORKSPACE / source_root
    source_split = source_root / args.split
    segments = selected_segments(
        manifest,
        args.split,
        args.task,
        args.max_episodes,
        args.max_episodes_per_task,
    )
    if not segments:
        raise ValueError("no segments selected")
    for segment in segments:
        validate_source_frame(source_split, segment)

    if args.output is None:
        output = PROJECT_DIR / "data" / f"calvin_d_10_{args.split}_{args.storage}"
    elif args.output.is_absolute():
        output = args.output
    else:
        # Treat CLI-relative outputs as workspace-relative, so invoking this
        # script from multitask_dit_policy/ or the repository root produces
        # the same dataset location.
        output = WORKSPACE / args.output
    output = output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
        shutil.rmtree(output)

    dataset = LeRobotDataset.create(
        repo_id=output.name,
        root=output,
        fps=int(manifest["frequency_hz"]),
        robot_type="calvin_panda",
        features=features(args.storage),
        use_videos=args.storage == "video",
        image_writer_threads=args.image_writer_threads,
        batch_encoding_size=1,
    )
    try:
        for segment in tqdm(segments, desc=f"Converting {args.split}/{args.storage}", unit="episode"):
            for frame in iter_episode_frames(source_split, segment):
                dataset.add_frame(frame)
            dataset.save_episode(parallel_encoding=False)
        dataset.finalize()
    except BaseException:
        # Flush/stop asynchronous writers before propagating the original error.
        if dataset.writer is not None:
            dataset.writer.stop_image_writer()
        raise

    conversion_info = {
        "output": str(output),
        "manifest": str(args.manifest.resolve()),
        "source_dataset": str(source_root),
        "split": args.split,
        "storage": args.storage,
        "episodes": len(segments),
        "frames": sum(segment["num_frames"] for segment in segments),
        "tasks": sorted({segment["calvin_task"] for segment in segments}),
    }
    (output / "conversion_info.json").write_text(json.dumps(conversion_info, indent=2) + "\n")
    print(json.dumps(conversion_info, indent=2))


if __name__ == "__main__":
    main()
