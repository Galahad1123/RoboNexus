#!/usr/bin/env python3
"""Audit clean CALVIN ABC demonstrations and build train/ABC-val/D-val manifests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from build_audited_manifests import (  # noqa: E402
    cluster_annotations,
    load_annotations,
    make_env,
    make_oracle,
    resolve,
)
from build_d34_manifests import audit_candidate as audit_general_candidate  # noqa: E402
from build_refined_manifests import (  # noqa: E402
    audit_rotate_candidate,
    audit_stack_candidate,
)


DEFAULT_CONFIG = WORKSPACE / "datasets/calvin_specialists/configs/calvin_abc_clean.json"


def load_scene_ranges(source_root: Path):
    raw = np.load(source_root / "training/scene_info.npy", allow_pickle=True).item()
    ranges = {}
    for key, value in raw.items():
        suffix = str(key).rsplit("_", 1)[-1]
        if suffix in {"A", "B", "C"}:
            ranges[suffix] = tuple(map(int, value))
    if set(ranges) != {"A", "B", "C"}:
        raise ValueError("training/scene_info.npy must define A, B, and C ranges")
    return ranges


def environment_for_row(row, source_split, ranges):
    if source_split == "validation":
        return "D"
    matches = [env for env, (start, end) in ranges.items() if start <= row["start"] and row["end"] <= end]
    if len(matches) != 1:
        raise ValueError("Interval [{}, {}] crosses or misses ABC scene boundaries".format(row["start"], row["end"]))
    return matches[0]


def audit_row(env, oracle, split_dir, row, dataset_cfg, config):
    task = row["calvin_task"]
    if task.startswith("rotate_"):
        cfg = dict(config["rotate_audit"])
        return audit_rotate_candidate(env, oracle, split_dir, row, cfg)
    if task == "stack_block":
        return audit_stack_candidate(env, oracle, split_dir, row, config["stack_audit"])
    return audit_general_candidate(
        env,
        oracle,
        split_dir,
        row,
        int(config["max_start_repair_frames"]),
        int(config["max_end_repair_frames"]),
    )


def retained_segment(row, split_name, source_split, task_id, instruction, cluster_id, environment):
    roles = row.get("mask_roles")
    if row["calvin_task"] == "stack_block" and row.get("source_object") and row.get("demonstrated_destination"):
        roles = {
            "source": [{"movable": row["source_object"]}],
            "destination": [{"movable": row["demonstrated_destination"]}],
        }
    return {
        "segment_id": "{}-{:06d}".format(split_name, row["annotation_index"]),
        "cluster_id": cluster_id,
        "split": split_name,
        "source_split": source_split,
        "environment": environment,
        "task_id": task_id,
        "calvin_task": row["calvin_task"],
        "instruction": instruction,
        "original_instruction": row["original_instruction"],
        "original_start": row.get("original_start", row["start"]),
        "original_end": row.get("original_end", row["end"]),
        "start": row["start"],
        "end": row["end"],
        "num_frames": row["end"] - row["start"] + 1,
        "oracle_consistent": True,
        "episode_variant": row.get("episode_variant"),
        "source_object": row.get("source_object"),
        "demonstrated_destination": row.get("demonstrated_destination"),
        "mask_roles": roles,
    }


def audit_source_split(dataset_name, dataset_cfg, source_split, source_root, oracle, config, ranges):
    split_dir = source_root / source_split
    tasks = dataset_cfg["tasks"]
    candidates = load_annotations(split_dir, set(tasks))
    clusters = cluster_annotations(candidates, float(config["overlap_threshold"]))
    task_order = sorted(tasks)
    env = make_env(split_dir)
    audited, retained, cluster_rows = [], [], []
    try:
        for cluster_index, cluster in enumerate(clusters):
            environment = environment_for_row(cluster[0], source_split, ranges)
            rows = [audit_row(env, oracle, split_dir, row, dataset_cfg, config) for row in cluster]
            for row in rows:
                row["environment"] = environment
                row["source_split"] = source_split
            audited.extend(rows)
            valid = [row for row in rows if row["oracle_consistent"]]
            representative = min(valid, key=lambda row: (row["num_frames"], row["annotation_index"])) if valid else None
            task = cluster[0]["calvin_task"]
            cluster_id = "{}-{}-{:04d}".format(environment, task, cluster_index)
            cluster_rows.append({
                "cluster_id": cluster_id,
                "environment": environment,
                "task": task,
                "candidate_annotation_indices": [row["annotation_index"] for row in rows],
                "oracle_consistent_candidates": [row["annotation_index"] for row in valid],
                "selected_annotation_index": representative["annotation_index"] if representative else None,
            })
            if representative:
                retained.append(retained_segment(
                    representative,
                    source_split,
                    source_split,
                    task_order.index(task),
                    tasks[task],
                    cluster_id,
                    environment,
                ))
    finally:
        env.close()
        env.ownsPhysicsClient = False
    return retained, audited, cluster_rows


def stable_score(seed, row):
    value = "{}:{}:{}:{}".format(seed, row["environment"], row["calvin_task"], row["cluster_id"])
    return hashlib.sha256(value.encode()).digest()


def split_abc(rows, ratio, seed):
    by_stratum = defaultdict(list)
    for row in rows:
        by_stratum[(row["environment"], row["calvin_task"])].append(row)
    training, validation = [], []
    for _, group in sorted(by_stratum.items()):
        group = sorted(group, key=lambda row: stable_score(seed, row))
        count = min(len(group) - 1, max(1, round(len(group) * ratio))) if len(group) > 1 else 0
        validation.extend(group[:count])
        training.extend(group[count:])
    for row in training:
        row["split"] = "training"
        row["segment_id"] = row["segment_id"].replace("training-", "training-", 1)
    for row in validation:
        row["split"] = "abc_validation"
        row["segment_id"] = row["segment_id"].replace("training-", "abc_validation-", 1)
    return training, validation


def summary(name, split, rows, audited=None):
    by_task = Counter(row["calvin_task"] for row in rows)
    by_environment = Counter(row["environment"] for row in rows)
    by_task_environment = Counter((row["calvin_task"], row["environment"]) for row in rows)
    rejection = Counter(
        reason for row in (audited or []) for reason in row.get("rejection_reasons", [])
    )
    return {
        "dataset": name,
        "split": split,
        "episodes": len(rows),
        "frames": sum(row["num_frames"] for row in rows),
        "by_task": dict(sorted(by_task.items())),
        "by_environment": dict(sorted(by_environment.items())),
        "by_task_environment": {
            task: {env: by_task_environment[(task, env)] for env in ("A", "B", "C", "D") if by_task_environment[(task, env)]}
            for task in sorted(by_task)
        },
        "candidate_rejection_reasons": dict(sorted(rejection.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", action="append")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    selected = args.dataset or sorted(config["datasets"])
    unknown = sorted(set(selected) - set(config["datasets"]))
    if unknown:
        parser.error("unknown datasets: {}".format(unknown))
    source_root = resolve(WORKSPACE, config["source_dataset"]).resolve()
    ranges = load_scene_ranges(source_root)
    oracle = make_oracle(resolve(WORKSPACE, config["oracle_config"]))
    all_summaries = {}
    for name in selected:
        cfg = config["datasets"][name]
        abc_rows, abc_audit, abc_clusters = audit_source_split(name, cfg, "training", source_root, oracle, config, ranges)
        d_rows, d_audit, d_clusters = audit_source_split(name, cfg, "validation", source_root, oracle, config, ranges)
        train_rows, abc_val_rows = split_abc(abc_rows, float(config["abc_validation_ratio"]), int(config["split_seed"]))
        for row in d_rows:
            row["split"] = "d_validation"
            row["segment_id"] = row["segment_id"].replace("validation-", "d_validation-", 1)
        split_rows = {"training": train_rows, "abc_validation": abc_val_rows, "d_validation": d_rows}
        summaries = {split: summary(name, split, rows) for split, rows in split_rows.items()}
        summaries["abc_audit"] = summary(name, "abc_audit", abc_rows, abc_audit)
        summaries["d_audit"] = summary(name, "d_audit", d_rows, d_audit)
        output = WORKSPACE / "datasets" / name
        (output / "manifests").mkdir(parents=True, exist_ok=True)
        (output / "reports").mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": 1,
            "name": name,
            "source_dataset": config["source_dataset"],
            "frequency_hz": config["frequency_hz"],
            "image_mode": "clean_rgb",
            "abc_validation_ratio": config["abc_validation_ratio"],
            "split_seed": config["split_seed"],
            "tasks": [{"task_id": i, "calvin_task": task, "instruction": cfg["tasks"][task]} for i, task in enumerate(sorted(cfg["tasks"]))],
            "splits": {
                split: {"source_split": "validation" if split == "d_validation" else "training", "summary": summaries[split], "segments": rows}
                for split, rows in split_rows.items()
            },
        }
        (output / "manifests/segments.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (output / "reports/abc_oracle_audit.json").write_text(json.dumps(abc_audit, indent=2) + "\n")
        (output / "reports/d_oracle_audit.json").write_text(json.dumps(d_audit, indent=2) + "\n")
        (output / "reports/abc_overlap_clusters.json").write_text(json.dumps(abc_clusters, indent=2) + "\n")
        (output / "reports/d_overlap_clusters.json").write_text(json.dumps(d_clusters, indent=2) + "\n")
        (output / "reports/summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        all_summaries[name] = summaries
    print(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()
