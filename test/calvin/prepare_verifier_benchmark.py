#!/usr/bin/env python3
"""Prepare a small Oracle-labelled verifier benchmark from CALVIN-D demos."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
from omegaconf import OmegaConf
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "calvin/dataset/task_ABC_D"
DEFAULT_TASKS = [
    "rotate_red_block_right",
    "rotate_blue_block_left",
]

SUCCESS_CRITERIA = {
    "open_drawer": [
        ("drawer_displaced", "The lower-right drawer with the gray U-shaped handle is pulled outward; its front is no longer flush with the cabinet.", "static", False),
        ("interior_visible", "A genuine gap plus the bottom or inner side walls of the lower-right drawer are visible; space around the U-shaped handle does not count.", "static", False),
        ("stable_open", "The drawer opening remains approximately unchanged across all recent and current observations.", "temporal", True),
    ],
    "close_drawer": [
        ("closer_to_closed_reference", "The current lower-right drawer is substantially closer to the TOP closed reference than to the BOTTOM open reference.", "static", False, ["closer to the open", "matches the open", "drawer is open", "not closed"]),
        ("opening_closed", "No substantial accessible drawer interior or opening remains visible.", "static", False, ["drawer is open", "interior is visible", "interior visible", "not closed"]),
    ],
    "move_slider_left": [
        ("slider_at_left", "The sliding door panel with the vertical gray handle is at the left side of its travel range. A panel around the table/cabinet middle is the RIGHT state and must fail this criterion.", "static", False),
        ("right_section_exposed", "The right cabinet section is visibly exposed by the moved sliding door.", "static", False),
        ("stable_left", "The sliding door position remains approximately unchanged across all recent and current observations.", "temporal", True),
    ],
    "move_slider_right": [
        ("slider_at_right", "The sliding door panel with the vertical gray handle is around the middle/right side of its travel range, which is the CALVIN RIGHT state; a panel remaining at the far left must fail.", "static", False),
        ("left_section_exposed", "The left cabinet/slider section is visibly exposed by the door moving right.", "static", False),
    ],
    "turn_on_lightbulb": [
        ("yellow_small_sphere_visible", "A small yellow sphere is directly visible near the top of CURRENT_STATIC_VIEW. Judge only this visual fact; if no yellow small sphere is visible, this criterion must fail.", "static", False),
    ],
    "turn_off_led": [
        ("green_absent_at_led", "No green rectangle is visible at the LED landmark above the slider; green means ON and a plain white rectangle means OFF.", "static", False),
        ("matches_off_reference", "The CURRENT cropped static image matches the plain-white LED state shown in the successful GOAL cropped static image rather than a green ON state.", "static", False),
    ],
    "turn_on_led": [
        ("green_rectangle_visible", "A green rectangle is directly visible in the upper part of CURRENT_STATIC_VIEW. If present, pass; if absent, fail so the task remains in progress.", "static", False),
    ],
    "turn_off_lightbulb": [
        ("yellow_circle_absent", "Judge only whether a yellow circular object is visible in the upper part of CURRENT_STATIC_VIEW. If yellow circle is present, fail so the task remains in progress; if absent, pass.", "static", False),
    ],
    "lift_red_block_table": [
        ("red_cube_grasped", "First verify that a red cube is directly visible in CURRENT_WRIST_VIEW, then compare with GOAL_REFERENCE_WRIST_VIEW. Pass only if the current gripper visibly holds that red cube between its fingers in the reference-like holding relation; an empty current wrist must fail regardless of the goal image.", "wrist", False),
        ("red_cube_off_table", "In CURRENT_STATIC_VIEW, the red cube is elevated with the gripper and is not visibly remaining on the tabletop. A red cube visibly resting on the table must fail.", "static", False),
    ],
    "place_in_drawer": [
        ("block_inside_drawer", "The previously held block is visibly supported inside the drawer.", "static", False),
        ("block_released", "The block is separated from and no longer retained by the gripper.", "both", False),
        ("placement_stable", "The block remains inside the drawer without substantial motion across all recent and current observations.", "temporal", True),
    ],
    "place_in_slider": [
        ("rectangular_object_placed_on_table", "In CURRENT_STATIC_VIEW, the manipulated rectangular object is visibly placed and supported on a table surface rather than being carried in the air by the robot.", "static", False),
        ("gripper_not_holding_rectangular_object", "Compare CURRENT_WRIST_VIEW with the successful released GOAL_REFERENCE_WRIST_VIEW. The current gripper is not holding any rectangular object between its fingers; a visibly grasped rectangular object must fail.", "wrist", False),
    ],
    "stack_block": [
        ("vertical_stack", "One block is visibly positioned on top of another block with overlapping support.", "static", False),
        ("upper_block_released", "The upper block is separated from and no longer retained by the gripper.", "both", False),
        ("stack_stable", "The two-block stack remains upright and approximately unchanged across all recent and current observations.", "temporal", True),
    ],
    "unstack_block": [
        ("no_stack_in_current_static", "Using only CURRENT_STATIC_VIEW, no block/cube visibly overlaps and rests vertically on another block/cube. Any clear cube-on-cube stack must fail.", "static", False),
        ("no_stack_in_current_wrist", "Using only CURRENT_WRIST_VIEW, no block/cube visibly overlaps and rests vertically on another block/cube. Any clear cube-on-cube stack must fail.", "wrist", False),
    ],
    "push_into_drawer": [
        ("block_inside_drawer", "A block that started on the main tabletop has been pushed across the drawer boundary and is visibly inside the open lower-right drawer.", "static", False),
        ("block_not_on_main_table", "The pushed block is no longer resting at its initial location on the main tabletop outside the drawer.", "static", False),
        ("push_result_stable", "The block remains inside the drawer without substantial motion across recent and current observations.", "temporal", True),
    ],
}

ROTATE_TASKS = [
    "rotate_red_block_right", "rotate_red_block_left",
    "rotate_blue_block_right", "rotate_blue_block_left",
    "rotate_pink_block_right", "rotate_pink_block_left",
]
for rotate_task in ROTATE_TASKS:
    SUCCESS_CRITERIA[rotate_task] = [
        (
            "gripper_released_object",
            "Judge only CURRENT_WRIST_VIEW. The two thin pointed objects at the top are the "
            "two gripper fingers. Pass only when neither finger contacts a colored block/cube/"
            "rectangle; visible contact means not released and must fail. Ignore rotation.",
            "wrist",
            False,
        )
    ]


def criteria_for(task: str) -> List[Dict[str, Any]]:
    criteria = []
    for values in SUCCESS_CRITERIA[task]:
        criterion_id, description, evidence_source, temporal_required = values[:4]
        item = {
            "criterion_id": criterion_id,
            "description": description,
            "required": True,
            "evidence_source": evidence_source,
            "temporal_required": temporal_required,
        }
        if len(values) > 4:
            item["fail_if_evidence_contains"] = list(values[4])
        criteria.append(item)
    return criteria


def load_segments(dataset: Path, split: str) -> Dict[str, List[Dict[str, Any]]]:
    path = dataset / split / "lang_annotations/auto_lang_ann.npy"
    data = np.load(str(path), allow_pickle=True).item()
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for task, instruction, interval in zip(
        data["language"]["task"], data["language"]["ann"], data["info"]["indx"]
    ):
        start, end = map(int, interval)
        grouped[str(task)].append(
            {
                "task": str(task),
                "instruction": str(instruction),
                "start": start,
                "end": end,
                "length": end - start + 1,
            }
        )
    return grouped


def median_segment(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    median = float(np.median([item["length"] for item in segments]))
    return min(segments, key=lambda item: (abs(item["length"] - median), item["start"]))


def distinct_reference_segment(
    segments: List[Dict[str, Any]], selected: Dict[str, Any]
) -> Dict[str, Any]:
    """Choose a representative completion from a different demonstration."""

    candidates = [
        item
        for item in segments
        if item["end"] < selected["start"] or item["start"] > selected["end"]
    ]
    if not candidates:
        raise ValueError(
            "task {} has no non-overlapping reference demonstration".format(selected["task"])
        )
    median = float(np.median([item["length"] for item in candidates]))
    return min(candidates, key=lambda item: (abs(item["length"] - median), item["start"]))


def empty_drawer_reference_segment(
    segments: List[Dict[str, Any]],
    selected: Dict[str, Any],
    dataset: Path,
    split: str,
    env: Any,
) -> Dict[str, Any]:
    """Choose an open-drawer demo with empty closed/start and open/end states."""

    candidates = [
        item
        for item in segments
        if item["end"] < selected["start"] or item["start"] > selected["end"]
    ]
    median = float(np.median([item["length"] for item in candidates]))
    candidates.sort(key=lambda item: (abs(item["length"] - median), item["start"]))
    for item in candidates:
        infos = []
        for frame_id in (item["start"], item["end"]):
            frame = load_frame(dataset / split / "episode_{:07d}.npz".format(frame_id))
            infos.append(reset_info(env, frame))
        empty = []
        for info in infos:
            table = info["scene_info"]["fixed_objects"]["table"]
            table_uid = table["uid"]
            drawer_link = table["links"]["drawer_link"]
            contains_cube = any(
                any(
                    contact[2] == table_uid and contact[4] == drawer_link
                    for contact in movable["contacts"]
                )
                for movable in info["scene_info"]["movable_objects"].values()
            )
            empty.append(not contains_cube)
        closed_state = infos[0]["scene_info"]["doors"]["base__drawer"]["current_state"]
        open_state = infos[1]["scene_info"]["doors"]["base__drawer"]["current_state"]
        if all(empty) and closed_state < 0.08 and open_state > 0.12:
            return item
    raise ValueError("no non-overlapping empty open_drawer reference demo was found")


def make_env(dataset: Path, split: str):
    config = OmegaConf.load(dataset / split / ".hydra/merged_config.yaml")
    if "tactile" in config.cameras:
        del config.cameras["tactile"]
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    return hydra.utils.instantiate(
        config.env, show_gui=False, use_egl=True, use_vr=False, use_scene_info=True
    )


def make_oracle():
    config = OmegaConf.load(
        REPO_ROOT / "calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    return hydra.utils.instantiate(config)


def load_frame(path: Path) -> Dict[str, np.ndarray]:
    with np.load(str(path)) as frame:
        return {key: frame[key].copy() for key in ("robot_obs", "scene_obs", "rgb_static", "rgb_gripper")}


def reset_info(env: Any, frame: Dict[str, np.ndarray]) -> Dict[str, Any]:
    env.reset(robot_obs=frame["robot_obs"], scene_obs=frame["scene_obs"])
    return env.get_info()


def save_rgb(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def task_static_view(task: str, array: np.ndarray) -> np.ndarray:
    """Crop small fixed facilities and resize the crop back to the source size."""

    height, width = array.shape[:2]
    if task in {"turn_on_lightbulb", "turn_off_lightbulb"}:
        crop = array[: height // 2, width // 2 :]
    elif task in {"turn_on_led", "turn_off_led"}:
        crop = array[: height // 2, : width // 2]
    else:
        return array
    return np.asarray(
        Image.fromarray(crop.astype(np.uint8), mode="RGB").resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )
    )


def close_drawer_reference(closed: np.ndarray, opened: np.ndarray) -> np.ndarray:
    """TOP=closed lower half, BOTTOM=open lower half, preserving original size."""

    height = closed.shape[0]
    return np.concatenate((closed[height // 2 :], opened[height // 2 :]), axis=0)


def lower_half_pair(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Stack the lower halves of two frames while preserving source dimensions."""

    height = top.shape[0]
    return np.concatenate((top[height // 2 :], bottom[height // 2 :]), axis=0)


def oracle_validated_segment(
    segments: List[Dict[str, Any]], dataset: Path, split: str, env: Any, oracle: Any
) -> Dict[str, Any]:
    """Choose a median-like demonstration whose end state passes the Task Oracle."""

    median = float(np.median([item["length"] for item in segments]))
    candidates = sorted(
        segments, key=lambda item: (abs(item["length"] - median), item["start"])
    )
    for item in candidates:
        start = load_frame(dataset / split / "episode_{:07d}.npz".format(item["start"]))
        end = load_frame(dataset / split / "episode_{:07d}.npz".format(item["end"]))
        start_info = reset_info(env, start)
        end_info = reset_info(env, end)
        achieved = oracle.get_task_info_for_set(start_info, end_info, {item["task"]})
        if item["task"] in achieved:
            return item
    raise ValueError("no Oracle-positive demonstration found for {}".format(segments[0]["task"]))


def prepare(dataset: Path, split: str, output: Path, tasks: List[str]) -> Dict[str, Any]:
    grouped = load_segments(dataset, split)
    unknown = sorted(set(tasks) - set(grouped))
    if unknown:
        raise ValueError("tasks missing from CALVIN-D annotations: {}".format(unknown))
    env = make_env(dataset, split)
    oracle = make_oracle()
    selected = [
        oracle_validated_segment(grouped[task], dataset, split, env, oracle)
        for task in tasks
    ]
    cases = []
    try:
        for segment in selected:
            task = segment["task"]
            reference_segment = (
                empty_drawer_reference_segment(
                    grouped["open_drawer"], segment, dataset, split, env
                )
                if task == "close_drawer"
                else distinct_reference_segment(grouped[task], segment)
            )
            reference_path = dataset / split / "episode_{:07d}.npz".format(
                reference_segment["start"] if task == "close_drawer" else reference_segment["end"]
            )
            reference_frame = load_frame(reference_path)
            reference_rel = Path("assets") / (task + "__goal_reference__static.png")
            if task == "close_drawer":
                reference_open = load_frame(
                    dataset / split / "episode_{:07d}.npz".format(reference_segment["end"])
                )
                reference_static = close_drawer_reference(
                    reference_frame["rgb_static"], reference_open["rgb_static"]
                )
            else:
                reference_static = task_static_view(task, reference_frame["rgb_static"])
            save_rgb(reference_static, output / reference_rel)
            reference_wrist_rel = None
            if task.startswith("lift_") or task == "place_in_slider":
                reference_wrist_rel = Path("assets") / (task + "__goal_reference__wrist.png")
                save_rgb(reference_frame["rgb_gripper"], output / reference_wrist_rel)
            start_path = dataset / split / "episode_{:07d}.npz".format(segment["start"])
            end_path = dataset / split / "episode_{:07d}.npz".format(segment["end"])
            start_frame = load_frame(start_path)
            initial_static_rel = Path("assets") / (task + "__initial__static.png")
            initial_wrist_rel = Path("assets") / (task + "__initial__wrist.png")
            save_rgb(task_static_view(task, start_frame["rgb_static"]), output / initial_static_rel)
            save_rgb(start_frame["rgb_gripper"], output / initial_wrist_rel)
            start_info = reset_info(env, start_frame)
            negative_id = min(segment["start"] + 8, segment["end"] - 1)
            for phase, frame_id in (
                ("start_negative", negative_id),
                ("end_positive", segment["end"]),
            ):
                frame = load_frame(
                    dataset / split / "episode_{:07d}.npz".format(frame_id)
                )
                current_info = reset_info(env, frame)
                achieved = oracle.get_task_info_for_set(start_info, current_info, {task})
                case_id = "{}__{}".format(task, phase)
                static_rel = Path("assets") / (case_id + "__static.png")
                wrist_rel = Path("assets") / (case_id + "__wrist.png")
                save_rgb(task_static_view(task, frame["rgb_static"]), output / static_rel)
                save_rgb(frame["rgb_gripper"], output / wrist_rel)
                history_entries = []
                history_ids = sorted(
                    {
                        max(segment["start"], frame_id - 8),
                        max(segment["start"], frame_id - 4),
                    }
                )
                for history_index, history_id in enumerate(history_ids, 1):
                    if history_id >= frame_id:
                        continue
                    history_frame = load_frame(
                        dataset / split / "episode_{:07d}.npz".format(history_id)
                    )
                    history_static_rel = Path("assets") / (
                        case_id + "__history_{}__static.png".format(history_index)
                    )
                    history_wrist_rel = Path("assets") / (
                        case_id + "__history_{}__wrist.png".format(history_index)
                    )
                    save_rgb(
                        task_static_view(task, history_frame["rgb_static"]),
                        output / history_static_rel,
                    )
                    save_rgb(history_frame["rgb_gripper"], output / history_wrist_rel)
                    history_entries.append(
                        {
                            "frame_id": history_id,
                            "static_image": str(history_static_rel),
                            "wrist_image": str(history_wrist_rel),
                        }
                    )
                close_comparison = None
                if task == "close_drawer":
                    close_comparison_rel = Path("assets") / (
                        case_id + "__initial_top_current_bottom__static.png"
                    )
                    close_comparison = lower_half_pair(
                        start_frame["rgb_static"], frame["rgb_static"]
                    )
                    save_rgb(close_comparison, output / close_comparison_rel)
                cases.append(
                    {
                        "case_id": case_id,
                        "task": task,
                        "phase": phase,
                        "instruction": segment["instruction"],
                        "success_criteria": criteria_for(task),
                        "failure_conditions": [],
                        "segment": {key: segment[key] for key in ("start", "end", "length")},
                        "frame_id": frame_id,
                        "observation_context": {
                            "current_frame_id": frame_id,
                            "previous_frame_ids": history_ids,
                            "frame_interval_ms": 133,
                            "camera_views": ["static", "wrist"],
                            "visual_input_mode": (
                                "close_initial_current_and_goal_composites_static_only"
                                if task == "close_drawer"
                                else "upper_right_current_static_crop_only"
                                if task == "turn_on_lightbulb"
                                else "upper_left_current_static_crop_only"
                                if task == "turn_on_led"
                                else "upper_left_current_and_goal_static_crops_only"
                                if task == "turn_off_led"
                                else "upper_right_current_static_crop_only"
                                if task == "turn_off_lightbulb"
                                else "current_wrist_current_static_goal_wrist_only"
                                if task.startswith("lift_")
                                else "current_static_current_wrist_only"
                                if task == "unstack_block"
                                else "current_wrist_release_contact_only"
                                if task.startswith("rotate_")
                                else "current_static_current_wrist_goal_wrist_only"
                                if task == "place_in_slider"
                                else "current_static_goal_static_only"
                                if task == "move_slider_right"
                                else "full_static_and_wrist"
                            ),
                        },
                        "static_image": str(static_rel),
                        "wrist_image": str(wrist_rel),
                        "initial_static_image": str(initial_static_rel),
                        "initial_wrist_image": str(initial_wrist_rel),
                        "history": history_entries,
                        "close_initial_current_static": (
                            str(close_comparison_rel) if close_comparison is not None else None
                        ),
                        "close_recent_pair_static": None,
                        "goal_reference_static": str(reference_rel),
                        "goal_reference_wrist": (
                            str(reference_wrist_rel) if reference_wrist_rel is not None else None
                        ),
                        "reference_segment": {
                            key: reference_segment[key] for key in ("start", "end", "length")
                        },
                        "oracle_success": task in achieved,
                        "oracle_achieved_tasks": sorted(achieved),
                    }
                )
    finally:
        env.close()
        env.ownsPhysicsClient = False
    manifest = {
        "schema_version": 17,
        "dataset": str(dataset.resolve()),
        "split": split,
        "selection": "median-length demonstration per task",
        "tasks": tasks,
        "cases": cases,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", action="append", dest="tasks")
    args = parser.parse_args()
    manifest = prepare(args.dataset.resolve(), args.split, args.output_dir.resolve(), args.tasks or DEFAULT_TASKS)
    print(json.dumps({"tasks": len(manifest["tasks"]), "cases": len(manifest["cases"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
