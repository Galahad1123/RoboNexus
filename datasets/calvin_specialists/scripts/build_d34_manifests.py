#!/usr/bin/env python3
"""Audit CALVIN-D's 34 labels and build the Push-6 and General-21 manifests.

The existing Rotate-2 Release and Persistent Stack manifests remain immutable.
Run this script in the ``robonexus-calvin`` environment.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from build_audited_manifests import (
    WORKSPACE,
    cluster_annotations,
    held_objects,
    load_annotations,
    load_state,
    make_env,
    make_oracle,
    movable_contacts,
    resolve,
    supported_by,
    target_object_for_task,
)


DEFAULT_CONFIG = WORKSPACE / "datasets/calvin_specialists/configs/calvin_d34.json"
BUILDABLE = ("calvin_push_6", "calvin_general_21")


def movable_positions(info):
    return {
        name: np.asarray(obj["current_pos"], dtype=np.float64)
        for name, obj in info["scene_info"]["movable_objects"].items()
    }


def infer_push_into_source(start_info, end_info):
    candidates = []
    start_positions = movable_positions(start_info)
    end_positions = movable_positions(end_info)
    for name in sorted(start_positions):
        start_drawer = any(item["link"] == "drawer_link" for item in supported_by(start_info, name))
        end_drawer = any(item["link"] == "drawer_link" for item in supported_by(end_info, name))
        if end_drawer and not start_drawer:
            candidates.append(name)
    if len(candidates) == 1:
        return candidates[0]
    moved = sorted(
        (
            (float(np.linalg.norm(end_positions[name] - start_positions[name])), name)
            for name in start_positions
        ),
        reverse=True,
    )
    if moved and moved[0][0] >= 0.05 and (len(moved) == 1 or moved[0][0] > moved[1][0] + 0.02):
        return moved[0][1]
    return None


def infer_unstack_source(start_info):
    movable = start_info["scene_info"]["movable_objects"]
    candidates = []
    for name in movable:
        contacts = movable_contacts(start_info, name)
        if len(contacts) != 1 or supported_by(start_info, name):
            continue
        other = contacts[0]
        if float(movable[name]["current_pos"][2]) > float(movable[other]["current_pos"][2]):
            candidates.append(name)
    return candidates[0] if len(candidates) == 1 else None


def fixed_selector(task):
    if task in {"open_drawer", "close_drawer"}:
        return {"fixed": "table", "links": ["drawer_link"]}
    if task in {"move_slider_left", "move_slider_right"}:
        return {"fixed": "table", "links": ["slide_link"]}
    if task in {"turn_on_led", "turn_off_led"}:
        return {"fixed": "table", "links": ["button_link"]}
    if task in {"turn_on_lightbulb", "turn_off_lightbulb"}:
        return {"fixed": "table", "links": ["switch_link"]}
    return None


def audit_candidate(
    env, oracle, split_dir, row, max_start_repair_frames, max_end_repair_frames
):
    task = row["calvin_task"]
    start = row["start"]
    start_info = load_state(env, split_dir, start)
    if task.startswith("push_") and task != "push_into_drawer" and held_objects(start_info):
        for offset in range(1, max_start_repair_frames + 1):
            candidate_start = row["start"] - offset
            if candidate_start < 0:
                break
            candidate_info = load_state(env, split_dir, candidate_start)
            if not held_objects(candidate_info):
                start = candidate_start
                start_info = candidate_info
                break
    end = row["end"]
    end_info = load_state(env, split_dir, end)
    achieved = oracle.get_task_info_for_set(start_info, end_info, {task})
    if task not in achieved:
        for offset in range(1, max_end_repair_frames + 1):
            path = split_dir / "episode_{:07d}.npz".format(row["end"] + offset)
            if not path.is_file():
                break
            candidate_info = load_state(env, split_dir, row["end"] + offset)
            if task in oracle.get_task_info_for_set(start_info, candidate_info, {task}):
                end = row["end"] + offset
                end_info = candidate_info
                achieved = {task}
                break

    reasons = [] if task in achieved else ["oracle_not_satisfied"]
    initial_held = held_objects(start_info)
    source = None
    roles = None
    if task.startswith("push_") and task != "push_into_drawer":
        source = target_object_for_task(task)
        if initial_held:
            reasons.append("initially_holding_object")
        roles = {"source": [{"movable": source}]}
    elif task.startswith("lift_"):
        source = target_object_for_task(task)
        if initial_held:
            reasons.append("initially_holding_object")
        if held_objects(end_info) != [source]:
            reasons.append("lift_does_not_end_holding_target")
        roles = {"source": [{"movable": source}]}
    elif task in {"place_in_drawer", "place_in_slider"}:
        source = initial_held[0] if len(initial_held) == 1 else None
        if source is None:
            reasons.append("expected_exactly_one_held_object")
        link = "drawer_link" if task == "place_in_drawer" else "plank_link"
        roles = {"destination": [{"fixed": "table", "links": [link]}]}
    elif task == "push_into_drawer":
        if initial_held:
            reasons.append("initially_holding_object")
        source = infer_push_into_source(start_info, end_info)
        if source is None:
            reasons.append("ambiguous_pushed_object")
        else:
            roles = {
                "source": [{"movable": source}],
                "destination": [{"fixed": "table", "links": ["drawer_link"]}],
            }
    elif task == "unstack_block":
        if initial_held:
            reasons.append("initially_holding_object")
        source = infer_unstack_source(start_info)
        if source is None:
            reasons.append("ambiguous_unstack_source")
        else:
            roles = {"source": [{"movable": source}]}
    else:
        target = fixed_selector(task)
        if initial_held:
            reasons.append("initially_holding_object")
        if target is None:
            reasons.append("missing_mask_rule")
        else:
            roles = {"target": [target]}

    return {
        **row,
        "original_start": row["start"],
        "original_end": row["end"],
        "start": start,
        "end": end,
        "num_frames": end - start + 1,
        "start_extension_frames": row["start"] - start,
        "end_extension_frames": end - row["end"],
        "initial_held_objects": initial_held,
        "source_object": source,
        "oracle_consistent": not reasons,
        "rejection_reasons": reasons,
        "mask_roles": roles,
    }


def build_split(
    name,
    cfg,
    split,
    source_root,
    oracle,
    overlap_threshold,
    max_start_repair,
    max_end_repair,
):
    split_dir = source_root / split
    tasks = cfg["instructions"]
    candidates = load_annotations(split_dir, set(tasks))
    clusters = cluster_annotations(candidates, overlap_threshold)
    env = make_env(split_dir)
    audited, retained, cluster_rows = [], [], []
    try:
        for cluster_index, cluster in enumerate(clusters):
            rows = [
                audit_candidate(
                    env,
                    oracle,
                    split_dir,
                    row,
                    max_start_repair,
                    max_end_repair,
                )
                for row in cluster
            ]
            audited.extend(rows)
            valid = [row for row in rows if row["oracle_consistent"]]
            representative = min(valid, key=lambda row: (row["num_frames"], row["annotation_index"])) if valid else None
            task = cluster[0]["calvin_task"]
            cluster_id = "{}-{:04d}".format(task, cluster_index)
            cluster_rows.append({
                "cluster_id": cluster_id,
                "task": task,
                "candidate_annotation_indices": [row["annotation_index"] for row in rows],
                "oracle_consistent_candidates": [row["annotation_index"] for row in valid],
                "selected_annotation_index": representative["annotation_index"] if representative else None,
            })
            if representative is None:
                continue
            retained.append({
                "segment_id": "{}-{:06d}".format(split, representative["annotation_index"]),
                "cluster_id": cluster_id,
                "split": split,
                "task_id": sorted(tasks).index(task),
                "calvin_task": task,
                "instruction": tasks[task],
                "original_instruction": representative["original_instruction"],
                "original_start": representative["original_start"],
                "original_end": representative["original_end"],
                "start": representative["start"],
                "end": representative["end"],
                "num_frames": representative["num_frames"],
                "oracle_consistent": True,
                "source_object": representative["source_object"],
                "mask_roles": representative["mask_roles"],
            })
    finally:
        env.close()
        env.ownsPhysicsClient = False
    by_candidates = Counter(row["calvin_task"] for row in candidates)
    by_clusters = Counter(cluster[0]["calvin_task"] for cluster in clusters)
    by_retained = Counter(row["calvin_task"] for row in retained)
    rejection = Counter(reason for row in audited for reason in row["rejection_reasons"])
    summary = {
        "dataset": name,
        "split": split,
        "candidates": len(candidates),
        "clusters": len(clusters),
        "retained": len(retained),
        "frames_retained": sum(row["num_frames"] for row in retained),
        "by_task": {task: {"candidates": by_candidates[task], "clusters": by_clusters[task], "retained": by_retained[task]} for task in sorted(tasks)},
        "candidate_rejection_reasons": dict(sorted(rejection.items())),
    }
    return {"summary": summary, "segments": sorted(retained, key=lambda row: (row["task_id"], row["start"])), "audit": audited, "clusters": cluster_rows}


def write_inventory(config, source_root):
    routing = config["routing"]
    owners = {task: group for group, tasks in routing.items() for task in tasks}
    split_counts = {}
    observed = set()
    for split in ("training", "validation"):
        rows = load_annotations(source_root / split, set(owners))
        counts = Counter(row["calvin_task"] for row in rows)
        split_counts[split] = dict(sorted(counts.items()))
        observed.update(counts)
    if len(owners) != 34 or len(observed) != 34 or set(owners) != observed:
        raise ValueError("Routing must cover exactly the 34 observed CALVIN-D labels")
    report = {
        "version": 1,
        "source_dataset": config["source_dataset"],
        "num_tasks": 34,
        "routing": routing,
        "routing_is_disjoint": sum(len(tasks) for tasks in routing.values()) == len(owners),
        "annotation_counts": split_counts,
        "specialist_datasets": {
            "push": "calvin_push_6",
            "rotate": "calvin_rotate_2_release",
            "stack": "calvin_stack_pick_place_persistent",
            "general": "calvin_general_21",
        },
        "stack_auxiliary_training": config["stack_auxiliary_training"],
    }
    output = WORKSPACE / "datasets/calvin_d34/reports/task_inventory.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", action="append", choices=BUILDABLE)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source_root = resolve(WORKSPACE, config["source_dataset"]).resolve()
    inventory = write_inventory(config, source_root)
    if args.inventory_only:
        print(json.dumps(inventory, indent=2))
        return
    oracle = make_oracle(resolve(WORKSPACE, config["oracle_config"]))
    summaries = {}
    for name in args.dataset or BUILDABLE:
        cfg = config["datasets"][name]
        results = {
            split: build_split(
                name,
                cfg,
                split,
                source_root,
                oracle,
                float(config["overlap_threshold"]),
                int(config["max_start_repair_frames"]),
                int(config["max_end_repair_frames"]),
            )
            for split in ("training", "validation")
        }
        output = WORKSPACE / "datasets" / name
        (output / "manifests").mkdir(parents=True, exist_ok=True)
        (output / "reports").mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": 1,
            "name": name,
            "source_dataset": config["source_dataset"],
            "frequency_hz": config["frequency_hz"],
            "overlap_threshold": config["overlap_threshold"],
            "audit_profile": "d34_{}".format(cfg["group"]),
            "oracle_config": config["oracle_config"],
            "tasks": [{"task_id": index, "calvin_task": task, "instruction": cfg["instructions"][task]} for index, task in enumerate(sorted(cfg["instructions"]))],
            "splits": {split: {"summary": result["summary"], "segments": result["segments"]} for split, result in results.items()},
        }
        (output / "manifests/segments.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for split, result in results.items():
            (output / "reports/{}_oracle_audit.json".format(split)).write_text(json.dumps(result["audit"], indent=2) + "\n")
            (output / "reports/{}_overlap_clusters.json".format(split)).write_text(json.dumps(result["clusters"], indent=2) + "\n")
        summary = {split: result["summary"] for split, result in results.items()}
        (output / "reports/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries[name] = summary
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
