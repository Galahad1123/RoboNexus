#!/usr/bin/env python3
"""Build release-complete Rotate and mixed Stack+Pick+Place manifests.

This is the second-stage specialist audit. It is intentionally separate from
``build_audited_manifests.py`` so the datasets used by earlier experiments are
never rewritten. Run it in the ``robonexus-calvin`` environment.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from build_audited_manifests import (
    WORKSPACE,
    DEFAULT_CONFIG,
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


REFINED_DATASETS = ("calvin_rotate_6_release", "calvin_stack_pick_place")


def frame_arrays(split_dir, frame_index):
    with np.load(split_dir / "episode_{:07d}.npz".format(frame_index)) as frame:
        return {
            "robot_obs": frame["robot_obs"].copy(),
            "scene_obs": frame["scene_obs"].copy(),
            "rel_actions": frame["rel_actions"].copy(),
        }


def object_pose(info, name):
    obj = info["scene_info"]["movable_objects"][name]
    return np.asarray(obj["current_pos"]), np.asarray(obj["current_orn"])


def orientation_delta(left, right):
    return float(
        (Rotation.from_quat(right) * Rotation.from_quat(left).inv()).magnitude()
    )


def released_target(env, split_dir, frame_index, target):
    info = load_state(env, split_dir, frame_index)
    arrays = frame_arrays(split_dir, frame_index)
    return (
        float(arrays["robot_obs"][6]) >= 0.04
        and float(arrays["rel_actions"][-1]) > 0
        and bool(supported_by(info, target))
    ), info


def audit_rotate_candidate(env, oracle, split_dir, row, cfg):
    target = target_object_for_task(row["calvin_task"])
    effective_start = row["start"]
    start_info = load_state(env, split_dir, effective_start)
    if held_objects(start_info):
        for offset in range(1, int(cfg["max_start_repair_frames"]) + 1):
            candidate_start = row["start"] - offset
            if candidate_start < 0:
                break
            candidate_info = load_state(env, split_dir, candidate_start)
            if not held_objects(candidate_info):
                effective_start = candidate_start
                start_info = candidate_info
                break
    reasons = []
    if held_objects(start_info):
        reasons.append("initially_held")

    # CALVIN language intervals often end just before physical settling. Match
    # the established conservative boundary repair: oracle success and release
    # may occur at most 0.5 seconds after the annotation, then only a stable
    # post-release suffix may be retained.
    scan_limit = row["end"] + int(cfg["max_start_repair_frames"])
    oracle_frame = None
    release_run = []
    release_frame = None
    completion_frame = None
    infos = {}
    for frame_index in range(effective_start, scan_limit + 1):
        path = split_dir / "episode_{:07d}.npz".format(frame_index)
        if not path.is_file():
            break
        info = load_state(env, split_dir, frame_index)
        infos[frame_index] = info
        if oracle_frame is None:
            achieved = oracle.get_task_info_for_set(
                start_info, info, {row["calvin_task"]}
            )
            if row["calvin_task"] in achieved:
                oracle_frame = frame_index
        released, _ = released_target(env, split_dir, frame_index, target)
        if released:
            release_run.append(frame_index)
            if len(release_run) >= int(cfg["release_consecutive_frames"]):
                release_frame = release_run[-int(cfg["release_consecutive_frames"])]
        else:
            release_run = []
            release_frame = None
        if oracle_frame is not None and release_frame is not None:
            completion_frame = max(oracle_frame, release_frame)
            break

    if oracle_frame is None:
        reasons.append("oracle_not_satisfied")
    if release_frame is None:
        reasons.append("release_not_completed")

    suffix_end = completion_frame
    stable_suffix_frames = 0
    if completion_frame is not None:
        previous = infos.get(completion_frame) or load_state(
            env, split_dir, completion_frame
        )
        for frame_index in range(
            completion_frame + 1,
            completion_frame + int(cfg["stable_suffix_max_frames"]) + 1,
        ):
            path = split_dir / "episode_{:07d}.npz".format(frame_index)
            if not path.is_file():
                break
            released, current = released_target(env, split_dir, frame_index, target)
            previous_pos, previous_orn = object_pose(previous, target)
            current_pos, current_orn = object_pose(current, target)
            stable = (
                released
                and np.linalg.norm(current_pos - previous_pos)
                <= float(cfg["stable_position_delta_m"])
                and orientation_delta(previous_orn, current_orn)
                <= float(cfg["stable_orientation_delta_rad"])
            )
            if not stable:
                break
            suffix_end = frame_index
            stable_suffix_frames += 1
            previous = current
        if stable_suffix_frames < int(cfg["stable_suffix_min_frames"]):
            reasons.append("stable_release_suffix_too_short")

    return {
        **row,
        "original_start": row["start"],
        "original_end": row["end"],
        "start": effective_start,
        "end": suffix_end if suffix_end is not None else row["end"],
        "num_frames": (
            (suffix_end if suffix_end is not None else row["end"]) - effective_start + 1
        ),
        "start_extension_frames": row["start"] - effective_start,
        "oracle_success_frame": oracle_frame,
        "release_frame": release_frame,
        "stable_suffix_frames": stable_suffix_frames,
        "source_object": target,
        "episode_variant": "rotate_release_complete",
        "oracle_consistent": not reasons,
        "rejection_reasons": reasons,
        "mask_roles": {"target": [{"movable": target}]},
    }


def final_stack_pair(info):
    movable = info["scene_info"]["movable_objects"]
    pairs = []
    for source in movable:
        contacts = movable_contacts(info, source)
        fixed_support = supported_by(info, source)
        if len(contacts) != 1 or fixed_support:
            continue
        destination = contacts[0]
        source_z = float(movable[source]["current_pos"][2])
        destination_z = float(movable[destination]["current_pos"][2])
        if source_z > destination_z:
            pairs.append((source, destination))
    return pairs


def stable_grasp_start(env, split_dir, start, end, source, frames):
    run = []
    for frame_index in range(start, end + 1):
        info = load_state(env, split_dir, frame_index)
        if held_objects(info) == [source]:
            run.append(frame_index)
            if len(run) >= frames:
                return run[0]
        else:
            run = []
    return None


def audit_stack_candidate(env, oracle, split_dir, row, cfg):
    start_info = load_state(env, split_dir, row["start"])
    end_info = load_state(env, split_dir, row["end"])
    reasons = []
    achieved = oracle.get_task_info_for_set(start_info, end_info, {"stack_block"})
    if "stack_block" not in achieved:
        reasons.append("oracle_not_satisfied")
    end_arrays = frame_arrays(split_dir, row["end"])
    if not (
        float(end_arrays["rel_actions"][-1]) > 0
        and float(end_arrays["robot_obs"][6]) >= 0.04
    ):
        reasons.append("stack_does_not_end_released")
    pairs = final_stack_pair(end_info)
    if len(pairs) != 1:
        reasons.append("ambiguous_final_stack_pair")
        source = destination = None
    else:
        source, destination = pairs[0]
    grasp_start = None
    if source is not None:
        grasp_start = stable_grasp_start(
            env,
            split_dir,
            row["start"],
            row["end"],
            source,
            int(cfg["stable_grasp_frames"]),
        )
        if grasp_start is None:
            reasons.append("source_never_stably_grasped")

    contact_frames = 0
    if source is not None and destination is not None:
        for frame_index in range(max(row["start"], row["end"] - 4), row["end"] + 1):
            info = load_state(env, split_dir, frame_index)
            if destination in movable_contacts(info, source):
                contact_frames += 1
        if contact_frames < 3:
            reasons.append("stack_contact_not_persistent")

    initial_held = held_objects(start_info)
    return {
        **row,
        "original_start": row["start"],
        "original_end": row["end"],
        "start": row["start"],
        "end": row["end"],
        "num_frames": row["end"] - row["start"] + 1,
        "source_object": source,
        "demonstrated_destination": destination,
        "stable_grasp_start": grasp_start,
        "initial_held_objects": initial_held,
        "destination_contact_frames": contact_frames,
        "oracle_consistent": not reasons,
        "rejection_reasons": reasons,
    }


def audit_simple_candidate(env, oracle, split_dir, row):
    start_info = load_state(env, split_dir, row["start"])
    end_info = load_state(env, split_dir, row["end"])
    achieved = oracle.get_task_info_for_set(
        start_info, end_info, {row["calvin_task"]}
    )
    reasons = [] if row["calvin_task"] in achieved else ["oracle_not_satisfied"]
    task = row["calvin_task"]
    if task.startswith("lift_"):
        source = target_object_for_task(task)
        if held_objects(end_info) != [source]:
            reasons.append("pick_does_not_end_holding_target")
        variant = "table_pick"
        mask_roles = {
            "source": [{"movable": source, "when": "until_held"}]
        }
    else:
        held = held_objects(start_info)
        source = held[0] if len(held) == 1 else None
        if source is None:
            reasons.append("expected_exactly_one_held_object")
        variant = task
        link = "drawer_link" if task == "place_in_drawer" else "plank_link"
        mask_roles = {"destination": [{"fixed": "table", "links": [link]}]}
    return {
        **row,
        "original_start": row["start"],
        "original_end": row["end"],
        "start": row["start"],
        "end": row["end"],
        "num_frames": row["end"] - row["start"] + 1,
        "source_object": source,
        "episode_variant": variant,
        "oracle_consistent": not reasons,
        "rejection_reasons": reasons,
        "mask_roles": mask_roles,
    }


def stack_segments(representative, split, task_id, instruction, cluster_id):
    source = representative["source_object"]
    destination = representative["demonstrated_destination"]
    roles = {
        "source": [{"movable": source, "when": "until_held"}],
        "destination": [{"movable": destination}],
    }
    base = {
        "cluster_id": cluster_id,
        "split": split,
        "task_id": task_id,
        "calvin_task": "stack_block",
        "instruction": instruction,
        "original_instruction": representative["original_instruction"],
        "original_start": representative["original_start"],
        "original_end": representative["original_end"],
        "end": representative["end"],
        "oracle_consistent": True,
        "source_object": source,
        "demonstrated_destination": destination,
        "destination_contact_frames": representative["destination_contact_frames"],
        "mask_roles": roles,
    }
    result = []
    if not representative["initial_held_objects"]:
        result.append(
            {
                **base,
                "segment_id": "{}-{:06d}-full".format(
                    split, representative["annotation_index"]
                ),
                "episode_variant": "full_pick_and_stack",
                "start": representative["start"],
                "num_frames": representative["end"] - representative["start"] + 1,
            }
        )
    grasp_start = representative["stable_grasp_start"]
    result.append(
        {
            **base,
            "segment_id": "{}-{:06d}-held".format(
                split, representative["annotation_index"]
            ),
            "episode_variant": "held_start_stack",
            "start": grasp_start,
            "num_frames": representative["end"] - grasp_start + 1,
        }
    )
    return result


def evenly_spaced(rows, count):
    """Select a deterministic, timeline-spanning subset without front bias."""
    rows = sorted(rows, key=lambda row: (row["start"], row["segment_id"]))
    if count >= len(rows):
        return rows
    indices = [int((index + 0.5) * len(rows) / count) for index in range(count)]
    return [rows[index] for index in indices]


def apply_variant_caps(rows, caps):
    """Cap variants while keeping multi-color tasks approximately balanced."""
    if not caps:
        return rows, {}
    kept = [row for row in rows if row["episode_variant"] not in caps]
    report = {}
    for variant, cap in sorted(caps.items()):
        candidates = [row for row in rows if row["episode_variant"] == variant]
        by_task = {}
        for row in candidates:
            by_task.setdefault(row["calvin_task"], []).append(row)
        task_names = sorted(by_task)
        if not task_names:
            continue
        base, remainder = divmod(min(int(cap), len(candidates)), len(task_names))
        selected = []
        for index, task in enumerate(task_names):
            quota = base + (1 if index < remainder else 0)
            selected.extend(evenly_spaced(by_task[task], min(quota, len(by_task[task]))))
        # This fallback only matters if a small subgroup cannot fill its quota.
        if len(selected) < min(int(cap), len(candidates)):
            selected_ids = {row["segment_id"] for row in selected}
            remainder_rows = [
                row for row in candidates if row["segment_id"] not in selected_ids
            ]
            selected.extend(
                evenly_spaced(
                    remainder_rows,
                    min(int(cap), len(candidates)) - len(selected),
                )
            )
        kept.extend(selected)
        report[variant] = {
            "available": len(candidates),
            "cap": int(cap),
            "retained": len(selected),
        }
    return kept, report


def build_split(dataset_name, cfg, split, source_root, oracle, overlap_threshold):
    split_dir = source_root / split
    candidates = load_annotations(split_dir, set(cfg["tasks"]))
    clusters = cluster_annotations(candidates, overlap_threshold)
    task_order = sorted(cfg["tasks"])
    env = make_env(split_dir)
    audited, retained, cluster_report = [], [], []
    try:
        for cluster_index, cluster in enumerate(clusters):
            task = cluster[0]["calvin_task"]
            if cfg["audit_profile"] == "rotate_release":
                rows = [
                    audit_rotate_candidate(env, oracle, split_dir, row, cfg)
                    for row in cluster
                ]
            elif task == "stack_block":
                rows = [
                    audit_stack_candidate(env, oracle, split_dir, row, cfg)
                    for row in cluster
                ]
            else:
                rows = [
                    audit_simple_candidate(env, oracle, split_dir, row)
                    for row in cluster
                ]
            audited.extend(rows)
            valid = [row for row in rows if row["oracle_consistent"]]
            representative = (
                min(valid, key=lambda row: (row["num_frames"], row["annotation_index"]))
                if valid
                else None
            )
            cluster_id = "{}-{:04d}".format(task, cluster_index)
            cluster_report.append(
                {
                    "cluster_id": cluster_id,
                    "task": task,
                    "candidate_annotation_indices": [r["annotation_index"] for r in rows],
                    "oracle_consistent_candidates": [r["annotation_index"] for r in valid],
                    "selected_annotation_index": (
                        representative["annotation_index"] if representative else None
                    ),
                }
            )
            if representative is None:
                continue
            task_id = task_order.index(task)
            instruction = cfg["tasks"][task]
            if task == "stack_block":
                retained.extend(
                    stack_segments(representative, split, task_id, instruction, cluster_id)
                )
            else:
                retained.append(
                    {
                        "segment_id": "{}-{:06d}".format(
                            split, representative["annotation_index"]
                        ),
                        "cluster_id": cluster_id,
                        "split": split,
                        "task_id": task_id,
                        "calvin_task": task,
                        "instruction": instruction,
                        "original_instruction": representative["original_instruction"],
                        "original_start": representative["original_start"],
                        "original_end": representative["original_end"],
                        "start": representative["start"],
                        "end": representative["end"],
                        "num_frames": representative["num_frames"],
                        "oracle_consistent": True,
                        "episode_variant": representative["episode_variant"],
                        "source_object": representative["source_object"],
                        "oracle_success_frame": representative.get("oracle_success_frame"),
                        "release_frame": representative.get("release_frame"),
                        "stable_suffix_frames": representative.get("stable_suffix_frames"),
                        "mask_roles": representative["mask_roles"],
                    }
                )
    finally:
        env.close()
        env.ownsPhysicsClient = False

    variant_cap_report = {}
    if split == "training":
        retained, variant_cap_report = apply_variant_caps(
            retained, cfg.get("training_variant_caps")
        )

    by_task_candidates = Counter(row["calvin_task"] for row in candidates)
    by_task_clusters = Counter(cluster[0]["calvin_task"] for cluster in clusters)
    by_task_retained = Counter(row["calvin_task"] for row in retained)
    by_variant = Counter(row["episode_variant"] for row in retained)
    rejection_reasons = Counter(
        reason for row in audited for reason in row["rejection_reasons"]
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
                for task in task_order
            },
            "by_episode_variant": dict(sorted(by_variant.items())),
            "candidate_rejection_reasons": dict(sorted(rejection_reasons.items())),
            "variant_caps": variant_cap_report,
        },
        "segments": sorted(retained, key=lambda row: (row["task_id"], row["start"], row["segment_id"])),
        "audit": audited,
        "clusters": cluster_report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", action="append", choices=REFINED_DATASETS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    selected = args.dataset or list(REFINED_DATASETS)
    source_root = resolve(WORKSPACE, config["source_dataset"]).resolve()
    oracle = make_oracle(resolve(WORKSPACE, config["oracle_config"]))
    summaries = {}
    for dataset_name in selected:
        cfg = config["datasets"][dataset_name]
        output = WORKSPACE / "datasets" / dataset_name
        (output / "manifests").mkdir(parents=True, exist_ok=True)
        (output / "reports").mkdir(parents=True, exist_ok=True)
        results = {
            split: build_split(
                dataset_name,
                cfg,
                split,
                source_root,
                oracle,
                float(config["overlap_threshold"]),
            )
            for split in ("training", "validation")
        }
        manifest = {
            "version": 2,
            "name": dataset_name,
            "source_dataset": config["source_dataset"],
            "frequency_hz": config["frequency_hz"],
            "overlap_threshold": config["overlap_threshold"],
            "audit_profile": cfg["audit_profile"],
            "oracle_config": config["oracle_config"],
            "tasks": [
                {"task_id": i, "calvin_task": task, "instruction": cfg["tasks"][task]}
                for i, task in enumerate(sorted(cfg["tasks"]))
            ],
            "splits": {
                split: {
                    "summary": result["summary"],
                    "segments": result["segments"],
                }
                for split, result in results.items()
            },
        }
        (output / "manifests/segments.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        for split, result in results.items():
            (output / "reports/{}_oracle_audit.json".format(split)).write_text(
                json.dumps(result["audit"], indent=2) + "\n"
            )
            (output / "reports/{}_overlap_clusters.json".format(split)).write_text(
                json.dumps(result["clusters"], indent=2) + "\n"
            )
        summary = {split: result["summary"] for split, result in results.items()}
        (output / "reports/summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        summaries[dataset_name] = summary
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
