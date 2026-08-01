#!/usr/bin/env python3
"""Build deduplicated, CALVIN-oracle-consistent specialist manifests.

Run in the ``robonexus-calvin`` environment. Source NPZ files are read-only.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import hydra
import numpy as np
from omegaconf import OmegaConf


WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = WORKSPACE / "datasets/calvin_specialists/configs/datasets.json"


def load_annotations(split_dir: Path, requested_tasks: set) -> List[dict]:
    data = np.load(
        split_dir / "lang_annotations/auto_lang_ann.npy", allow_pickle=True
    ).item()
    rows = []
    for annotation_index, (task, instruction, interval) in enumerate(
        zip(
            data["language"]["task"],
            data["language"]["ann"],
            data["info"]["indx"],
        )
    ):
        task = str(task)
        if task not in requested_tasks:
            continue
        start, end = int(interval[0]), int(interval[1])
        rows.append(
            {
                "annotation_index": annotation_index,
                "calvin_task": task,
                "original_instruction": str(instruction),
                "start": start,
                "end": end,
                "num_frames": end - start + 1,
            }
        )
    return rows


def overlap_ratio(left: dict, right: dict) -> float:
    intersection = max(
        0, min(left["end"], right["end"]) - max(left["start"], right["start"]) + 1
    )
    return intersection / min(left["num_frames"], right["num_frames"])


def cluster_annotations(rows: List[dict], threshold: float) -> List[List[dict]]:
    """Connected components of same-task intervals above overlap threshold."""
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["calvin_task"]].append(row)
    result = []
    for task in sorted(by_task):
        task_rows = sorted(
            by_task[task], key=lambda row: (row["start"], row["end"], row["annotation_index"])
        )
        parents = list(range(len(task_rows)))

        def find(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left, right):
            left, right = find(left), find(right)
            if left != right:
                parents[right] = left

        for left_index, left in enumerate(task_rows):
            for right_index in range(left_index + 1, len(task_rows)):
                right = task_rows[right_index]
                if right["start"] > left["end"]:
                    break
                if overlap_ratio(left, right) >= threshold:
                    union(left_index, right_index)
        components = defaultdict(list)
        for index, row in enumerate(task_rows):
            components[find(index)].append(row)
        result.extend(
            sorted(component, key=lambda row: row["annotation_index"])
            for _, component in sorted(
                components.items(), key=lambda item: min(r["start"] for r in item[1])
            )
        )
    return result


def make_env(split_dir: Path):
    config = OmegaConf.load(split_dir / ".hydra/merged_config.yaml")
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


def make_oracle(path: Path):
    config = OmegaConf.load(path)
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    return hydra.utils.instantiate(config)


def held_objects(info: Dict[str, Any]) -> List[str]:
    robot_uid = int(info["robot_info"]["uid"])
    return sorted(
        name
        for name, obj in info["scene_info"]["movable_objects"].items()
        if any(int(contact[2]) == robot_uid for contact in obj["contacts"])
    )


def supported_by(info: Dict[str, Any], object_name: str) -> List[dict]:
    scene = info["scene_info"]
    obj = scene["movable_objects"][object_name]
    robot_uid = int(info["robot_info"]["uid"])
    fixed_by_uid = {
        int(fixed["uid"]): (name, fixed.get("links", {}))
        for name, fixed in scene["fixed_objects"].items()
    }
    supports = []
    for contact in obj["contacts"]:
        body_uid, link_index = int(contact[2]), int(contact[4])
        if body_uid == robot_uid or body_uid not in fixed_by_uid:
            continue
        body_name, links = fixed_by_uid[body_uid]
        link_name = next(
            (name for name, index in links.items() if int(index) == link_index),
            str(link_index),
        )
        identity = {"body": body_name, "link": link_name}
        if identity not in supports:
            supports.append(identity)
    return supports


def movable_contacts(info: Dict[str, Any], object_name: str) -> List[str]:
    scene = info["scene_info"]
    uid_to_name = {
        int(obj["uid"]): name for name, obj in scene["movable_objects"].items()
    }
    return sorted(
        {
            uid_to_name[int(contact[2])]
            for contact in scene["movable_objects"][object_name]["contacts"]
            if int(contact[2]) in uid_to_name
            and uid_to_name[int(contact[2])] != object_name
        }
    )


def target_object_for_task(task: str) -> str:
    for color in ("red", "blue", "pink"):
        if "_{}_block_".format(color) in task:
            return "block_{}".format(color)
    raise ValueError("Could not infer target object from {}".format(task))


def load_state(env, split_dir: Path, frame_index: int):
    with np.load(split_dir / "episode_{:07d}.npz".format(frame_index)) as frame:
        env.reset(robot_obs=frame["robot_obs"], scene_obs=frame["scene_obs"])
    env.p.performCollisionDetection(physicsClientId=env.cid)
    return env.get_info()


def audit_candidate(
    env,
    oracle,
    split_dir: Path,
    row: dict,
    max_boundary_repair_frames: int,
    stack_contact_window_frames: int,
    stack_min_persistent_contact_frames: int,
) -> dict:
    task = row["calvin_task"]
    original_start, original_end = row["start"], row["end"]
    effective_start = original_start
    start_info = load_state(env, split_dir, effective_start)
    start_held = held_objects(start_info)
    if task.startswith(("push_", "rotate_")) and start_held:
        for offset in range(1, max_boundary_repair_frames + 1):
            candidate_start = original_start - offset
            if candidate_start < 0:
                break
            candidate_info = load_state(env, split_dir, candidate_start)
            if not held_objects(candidate_info):
                effective_start = candidate_start
                start_info = candidate_info
                start_held = []
                break
    start_support = None
    if task.startswith(("push_", "rotate_")):
        start_support = supported_by(start_info, target_object_for_task(task))
    effective_end = original_end
    end_info = load_state(env, split_dir, effective_end)
    achieved = oracle.get_task_info_for_set(start_info, end_info, {task})
    if task not in achieved:
        for offset in range(1, max_boundary_repair_frames + 1):
            candidate_end = original_end + offset
            candidate_path = split_dir / "episode_{:07d}.npz".format(candidate_end)
            if not candidate_path.is_file():
                break
            candidate_info = load_state(env, split_dir, candidate_end)
            candidate_achieved = oracle.get_task_info_for_set(
                start_info, candidate_info, {task}
            )
            if task in candidate_achieved:
                effective_end = candidate_end
                end_info = candidate_info
                achieved = candidate_achieved
                break
    reasons = []
    if task.startswith(("push_", "rotate_")) and start_held:
        reasons.append("initially_held_after_boundary_repair")
    if task in {"stack_block", "place_in_drawer", "place_in_slider"} and len(start_held) != 1:
        reasons.append("expected_exactly_one_held_object")
    if task not in achieved:
        reasons.append("oracle_not_satisfied_after_boundary_repair")

    demonstrated_destination = None
    feasible_destinations = None
    final_movable_contacts = None
    destination_contact_frames = None
    if task == "stack_block" and len(start_held) == 1:
        source_object = start_held[0]
        feasible_destinations = sorted(
            set(start_info["scene_info"]["movable_objects"]) - {source_object}
        )
        final_movable_contacts = movable_contacts(end_info, source_object)
        if len(final_movable_contacts) == 1:
            demonstrated_destination = final_movable_contacts[0]
            destination_contact_frames = 0
            window_start = max(
                effective_start,
                effective_end - stack_contact_window_frames + 1,
            )
            for frame_index in range(window_start, effective_end + 1):
                frame_info = load_state(env, split_dir, frame_index)
                if demonstrated_destination in movable_contacts(
                    frame_info, source_object
                ):
                    destination_contact_frames += 1
            if destination_contact_frames < stack_min_persistent_contact_frames:
                reasons.append("stack_contact_not_persistent")
        else:
            reasons.append("ambiguous_final_movable_contact")
    else:
        source_object = start_held[0] if len(start_held) == 1 else None

    return {
        **row,
        "original_start": original_start,
        "original_end": original_end,
        "start": effective_start,
        "end": effective_end,
        "num_frames": effective_end - effective_start + 1,
        "start_extension_frames": original_start - effective_start,
        "end_extension_frames": effective_end - original_end,
        "oracle_consistent": not reasons,
        "rejection_reasons": reasons,
        "initial_held_objects": start_held,
        "initial_support": start_support,
        "source_object": source_object,
        "feasible_destinations": feasible_destinations,
        "demonstrated_destination": demonstrated_destination,
        "final_movable_contacts": final_movable_contacts,
        "destination_contact_frames": destination_contact_frames,
    }


def mask_roles_for(row: dict) -> dict:
    task = row["calvin_task"]
    if task.startswith(("push_", "rotate_")):
        return {"target": [{"movable": target_object_for_task(task)}]}
    if task == "stack_block":
        return {
            "destination": [
                {"movable": row["demonstrated_destination"]}
            ]
        }
    if task == "place_in_drawer":
        return {"destination": [{"fixed": "table", "links": ["drawer_link"]}]}
    if task == "place_in_slider":
        return {"destination": [{"fixed": "table", "links": ["plank_link"]}]}
    raise ValueError("No mask roles for {}".format(task))


def build_dataset_split(
    dataset_name: str,
    dataset_config: dict,
    split: str,
    source_root: Path,
    oracle,
    overlap_threshold: float,
    max_boundary_repair_frames: int,
    stack_contact_window_frames: int,
    stack_min_persistent_contact_frames: int,
) -> dict:
    split_dir = source_root / split
    tasks = dataset_config["tasks"]
    candidates = load_annotations(split_dir, set(tasks))
    clusters = cluster_annotations(candidates, overlap_threshold)
    env = make_env(split_dir)
    audited, retained, cluster_report = [], [], []
    try:
        for cluster_index, cluster in enumerate(clusters):
            audited_cluster = [
                audit_candidate(
                    env,
                    oracle,
                    split_dir,
                    row,
                    max_boundary_repair_frames,
                    stack_contact_window_frames,
                    stack_min_persistent_contact_frames,
                )
                for row in cluster
            ]
            audited.extend(audited_cluster)
            valid = [row for row in audited_cluster if row["oracle_consistent"]]
            representative = min(
                valid,
                key=lambda row: (row["num_frames"], row["start"], row["annotation_index"]),
            ) if valid else None
            cluster_id = "{}-{:04d}".format(cluster[0]["calvin_task"], cluster_index)
            cluster_report.append(
                {
                    "cluster_id": cluster_id,
                    "task": cluster[0]["calvin_task"],
                    "candidate_annotation_indices": [
                        row["annotation_index"] for row in audited_cluster
                    ],
                    "oracle_consistent_candidates": [
                        row["annotation_index"] for row in valid
                    ],
                    "selected_annotation_index": (
                        representative["annotation_index"] if representative else None
                    ),
                }
            )
            if representative is None:
                continue
            segment_id = "{}-{:06d}".format(split, representative["annotation_index"])
            retained.append(
                {
                    "segment_id": segment_id,
                    "cluster_id": cluster_id,
                    "split": split,
                    "task_id": sorted(tasks).index(representative["calvin_task"]),
                    "calvin_task": representative["calvin_task"],
                    "instruction": tasks[representative["calvin_task"]],
                    "original_instruction": representative["original_instruction"],
                    "original_start": representative["original_start"],
                    "original_end": representative["original_end"],
                    "start": representative["start"],
                    "end": representative["end"],
                    "num_frames": representative["num_frames"],
                    "start_extension_frames": representative["start_extension_frames"],
                    "end_extension_frames": representative["end_extension_frames"],
                    "oracle_consistent": True,
                    "initial_held_objects": representative["initial_held_objects"],
                    "initial_support": representative["initial_support"],
                    "source_object": representative["source_object"],
                    "feasible_destinations": representative["feasible_destinations"],
                    "demonstrated_destination": representative["demonstrated_destination"],
                    "destination_contact_frames": representative["destination_contact_frames"],
                    "mask_roles": mask_roles_for(representative),
                }
            )
    finally:
        env.close()
        env.ownsPhysicsClient = False

    by_task_candidates = Counter(row["calvin_task"] for row in candidates)
    by_task_clusters = Counter(cluster[0]["calvin_task"] for cluster in clusters)
    by_task_retained = Counter(row["calvin_task"] for row in retained)
    rejection_reasons = Counter(
        reason for row in audited for reason in row["rejection_reasons"]
    )
    support_distribution = Counter()
    for row in retained:
        support = row["initial_support"]
        if support is None:
            continue
        key = "+".join(
            "{}:{}".format(item["body"], item["link"]) for item in support
        ) or "no_fixed_support"
        support_distribution["{}|{}".format(row["calvin_task"], key)] += 1
    stack_pairs = Counter(
        "{}->{}".format(row["source_object"], row["demonstrated_destination"])
        for row in retained
        if row["calvin_task"] == "stack_block"
    )
    return {
        "summary": {
            "dataset": dataset_name,
            "split": split,
            "candidates": len(candidates),
            "clusters": len(clusters),
            "retained": len(retained),
            "frames_retained": sum(row["num_frames"] for row in retained),
            "by_task": {
                task: {
                    "candidates": by_task_candidates[task],
                    "clusters": by_task_clusters[task],
                    "retained": by_task_retained[task],
                }
                for task in sorted(tasks)
            },
            "candidate_rejection_reasons": dict(sorted(rejection_reasons.items())),
            "boundary_repairs": {
                "start_extended": sum(row["start_extension_frames"] > 0 for row in retained),
                "end_extended": sum(row["end_extension_frames"] > 0 for row in retained),
            },
            "initial_support_distribution": dict(sorted(support_distribution.items())),
            "stack_source_destination": dict(sorted(stack_pairs.items())),
        },
        "segments": sorted(retained, key=lambda row: (row["task_id"], row["start"])),
        "audit": audited,
        "clusters": cluster_report,
    }


def resolve(workspace: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else workspace / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", action="append", help="repeat to build selected datasets")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    source_root = resolve(WORKSPACE, config["source_dataset"]).resolve()
    oracle = make_oracle(resolve(WORKSPACE, config["oracle_config"]))
    selected = args.dataset or list(config["datasets"])
    unknown = sorted(set(selected) - set(config["datasets"]))
    if unknown:
        raise ValueError("Unknown datasets: {}".format(unknown))

    all_summaries = {}
    for dataset_name in selected:
        dataset_config = config["datasets"][dataset_name]
        output_root = WORKSPACE / "datasets" / dataset_name
        (output_root / "manifests").mkdir(parents=True, exist_ok=True)
        (output_root / "reports").mkdir(parents=True, exist_ok=True)
        split_results = {
            split: build_dataset_split(
                dataset_name,
                dataset_config,
                split,
                source_root,
                oracle,
                float(config["overlap_threshold"]),
                int(config["max_boundary_repair_frames"]),
                int(config["stack_contact_window_frames"]),
                int(config["stack_min_persistent_contact_frames"]),
            )
            for split in ("training", "validation")
        }
        manifest = {
            "version": 1,
            "name": dataset_name,
            "source_dataset": config["source_dataset"],
            "frequency_hz": config["frequency_hz"],
            "overlap_threshold": config["overlap_threshold"],
            "max_boundary_repair_frames": config["max_boundary_repair_frames"],
            "stack_contact_window_frames": config["stack_contact_window_frames"],
            "stack_min_persistent_contact_frames": config[
                "stack_min_persistent_contact_frames"
            ],
            "oracle_config": config["oracle_config"],
            "tasks": [
                {"task_id": index, "calvin_task": task, "instruction": instruction}
                for index, (task, instruction) in enumerate(sorted(dataset_config["tasks"].items()))
            ],
            "splits": {
                split: {
                    "summary": result["summary"],
                    "segments": result["segments"],
                }
                for split, result in split_results.items()
            },
        }
        (output_root / "manifests/segments.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        for split, result in split_results.items():
            (output_root / "reports/{}_oracle_audit.json".format(split)).write_text(
                json.dumps(result["audit"], indent=2) + "\n"
            )
            (output_root / "reports/{}_overlap_clusters.json".format(split)).write_text(
                json.dumps(result["clusters"], indent=2) + "\n"
            )
        summary = {split: result["summary"] for split, result in split_results.items()}
        (output_root / "reports/summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        all_summaries[dataset_name] = summary
    print(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()
