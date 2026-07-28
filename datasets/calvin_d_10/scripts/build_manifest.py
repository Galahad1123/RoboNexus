#!/usr/bin/env python3
"""Build a filtered CALVIN-D-10 segment manifest.

Run this script in the Python 3.8 ``robonexus-calvin`` environment. It restores
each candidate first frame in Bullet so contact-based filters are based on the
same state that closed-loop evaluation will use.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
from omegaconf import OmegaConf


WORKSPACE = Path(__file__).resolve().parents[3]
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "tasks.json"
DEFAULT_OUTPUT = PROJECT_DIR / "manifests" / "segments.json"


def make_state_env(split_dir: Path):
    config = OmegaConf.load(split_dir / ".hydra" / "merged_config.yaml")
    config.cameras.clear()
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    return hydra.utils.instantiate(
        config.env,
        show_gui=False,
        use_egl=False,
        use_vr=False,
        use_scene_info=True,
    )


def held_objects(info: Dict[str, Any]) -> List[str]:
    robot_uid = info["robot_info"]["uid"]
    return sorted(
        name
        for name, obj_info in info["scene_info"]["movable_objects"].items()
        if robot_uid in {contact[2] for contact in obj_info["contacts"]}
    )


def check_filters(filters: List[str], info: Dict[str, Any], thresholds: Dict[str, float]) -> List[str]:
    held = held_objects(info)
    doors = info["scene_info"]["doors"]
    lights = info["scene_info"]["lights"]
    failed = []
    for name in filters:
        passed = {
            "no_held_object": len(held) == 0,
            "exactly_one_held_object": len(held) == 1,
            "drawer_closed": doors["base__drawer"]["current_state"] <= thresholds["drawer_closed_max"],
            "drawer_open": doors["base__drawer"]["current_state"] >= thresholds["drawer_open_min"],
            "slider_right": doors["base__slide"]["current_state"] <= thresholds["slider_right_max"],
            "led_off": lights["led"]["logical_state"] <= thresholds["led_off_max"],
        }.get(name)
        if passed is None:
            raise ValueError(f"unknown initial-state filter: {name}")
        if not passed:
            failed.append(name)
    return failed


def load_annotations(split_dir: Path) -> List[dict]:
    path = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    data = np.load(path, allow_pickle=True).item()
    return [
        {
            "calvin_task": str(task),
            "original_instruction": str(instruction),
            "start": int(interval[0]),
            "end": int(interval[1]),
        }
        for task, instruction, interval in zip(
            data["language"]["task"], data["language"]["ann"], data["info"]["indx"]
        )
    ]


def build_split(
    source_root: Path, split: str, task_by_name: Dict[str, dict], thresholds: Dict[str, float]
) -> Dict[str, Any]:
    split_dir = source_root / split
    candidates = [row for row in load_annotations(split_dir) if row["calvin_task"] in task_by_name]
    env = make_state_env(split_dir)
    retained, rejected = [], []
    try:
        for index, row in enumerate(candidates):
            task = task_by_name[row["calvin_task"]]
            frame_path = split_dir / f"episode_{row['start']:07d}.npz"
            with np.load(frame_path) as frame:
                env.reset(robot_obs=frame["robot_obs"], scene_obs=frame["scene_obs"])
            info = env.get_info()
            held = held_objects(info)
            failed_filters = check_filters(task["initial_state_filters"], info, thresholds)
            segment = {
                "segment_id": f"{split}-{index:06d}",
                "split": split,
                "task_id": task["task_id"],
                "calvin_task": row["calvin_task"],
                "instruction": task["instruction"],
                "original_instruction": row["original_instruction"],
                "start": row["start"],
                "end": row["end"],
                "num_frames": row["end"] - row["start"] + 1,
                "initial_state": {
                    "held_objects": held,
                    "drawer": float(info["scene_info"]["doors"]["base__drawer"]["current_state"]),
                    "slider": float(info["scene_info"]["doors"]["base__slide"]["current_state"]),
                    "led": int(info["scene_info"]["lights"]["led"]["logical_state"]),
                },
            }
            if failed_filters:
                segment["failed_filters"] = failed_filters
                rejected.append(segment)
            else:
                retained.append(segment)
    finally:
        env.close()
        env.ownsPhysicsClient = False

    by_task_candidates = Counter(row["calvin_task"] for row in candidates)
    by_task_retained = Counter(row["calvin_task"] for row in retained)
    rejected_reasons = Counter(reason for row in rejected for reason in row["failed_filters"])
    return {
        "summary": {
            "candidates": len(candidates),
            "retained": len(retained),
            "rejected": len(rejected),
            "frames_retained": sum(row["num_frames"] for row in retained),
            "by_task": {
                task: {
                    "candidates": by_task_candidates[task],
                    "retained": by_task_retained[task],
                    "rejected": by_task_candidates[task] - by_task_retained[task],
                }
                for task in task_by_name
            },
            "rejected_reasons": dict(sorted(rejected_reasons.items())),
        },
        "segments": retained,
        "rejected_segments": rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", type=Path, help="override source_dataset from the config")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    source_root = (args.source or WORKSPACE / config["source_dataset"]).resolve()
    task_by_name = {task["calvin_task"]: task for task in config["tasks"]}
    splits = {
        split: build_split(source_root, split, task_by_name, config["filter_thresholds"])
        for split in ("training", "validation")
    }
    manifest = {
        "version": 1,
        "source_dataset": config["source_dataset"] if args.source is None else str(source_root),
        "task_config": str(args.config.resolve().relative_to(WORKSPACE)),
        "frequency_hz": config["frequency_hz"],
        "episode_boundary": config["episode_boundary"],
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({split: value["summary"] for split, value in splits.items()}, indent=2))


if __name__ == "__main__":
    main()
