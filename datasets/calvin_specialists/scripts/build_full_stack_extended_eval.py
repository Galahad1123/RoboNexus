#!/usr/bin/env python3
"""Build a validation-only Full Pick-and-Stack extended evaluation set.

Held-start validation demonstrations are backtracked within their original
continuous trajectory to a supported, no-object-held state. The resulting set
is reported separately from official validation and never used for training.
"""

import argparse
import json
from pathlib import Path

from build_audited_manifests import (
    WORKSPACE,
    held_objects,
    load_state,
    make_env,
    make_oracle,
    resolve,
    supported_by,
)


SOURCE_DATASET = "calvin_stack_pick_place_persistent"
OUTPUT_DATASET = "calvin_full_stack_extended_eval"
DEFAULT_ORACLE = "calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml"


def find_backtracked_start(env, oracle, split_dir, row, lookback_frames):
    source = row["source_object"]
    end_info = load_state(env, split_dir, row["end"])
    valid = []
    for frame_index in range(row["start"] - 1, max(-1, row["start"] - lookback_frames - 1), -1):
        path = split_dir / "episode_{:07d}.npz".format(frame_index)
        if not path.is_file():
            break
        info = load_state(env, split_dir, frame_index)
        if held_objects(info):
            if valid:
                break
            continue
        if not supported_by(info, source):
            if valid:
                break
            continue
        achieved = oracle.get_task_info_for_set(info, end_info, {"stack_block"})
        if "stack_block" not in achieved:
            if valid:
                break
            continue
        valid.append(frame_index)
    return min(valid) if valid else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-frames", type=int, default=90)
    parser.add_argument("--source-dataset", default=SOURCE_DATASET)
    parser.add_argument("--output-dataset", default=OUTPUT_DATASET)
    args = parser.parse_args()

    source_root = WORKSPACE / "datasets" / args.source_dataset
    source_manifest = json.loads((source_root / "manifests/segments.json").read_text())
    source_split = resolve(WORKSPACE, source_manifest["source_dataset"]) / "validation"
    oracle = make_oracle(resolve(WORKSPACE, source_manifest.get("oracle_config", DEFAULT_ORACLE)))
    env = make_env(source_split)
    rows, rejected = [], []
    try:
        stack_rows = [
            row
            for row in source_manifest["splits"]["validation"]["segments"]
            if row["calvin_task"] == "stack_block"
        ]
        for row in stack_rows:
            if row["episode_variant"] == "full_pick_and_stack":
                start = row["start"]
                origin = "official_full"
            else:
                start = find_backtracked_start(
                    env, oracle, source_split, row, args.lookback_frames
                )
                origin = "backtracked_from_held"
            if start is None:
                rejected.append({
                    "source_segment_id": row["segment_id"],
                    "reason": "no_valid_supported_unheld_prefix",
                })
                continue
            segment_id = "validation-full-ext-{:06d}".format(row["original_start"])
            rows.append({
                **row,
                "segment_id": segment_id,
                "start": start,
                "num_frames": row["end"] - start + 1,
                "episode_variant": "full_pick_and_stack_extended",
                "evaluation_tier": "official" if origin == "official_full" else "extended_backtracked",
                "extension_origin": origin,
                "source_segment_id": row["segment_id"],
                "backtrack_frames": row["start"] - start,
                "training_eligible": False,
            })
    finally:
        env.close()
        env.ownsPhysicsClient = False

    rows.sort(key=lambda row: (row["start"], row["segment_id"]))
    output = WORKSPACE / "datasets" / args.output_dataset
    (output / "manifests").mkdir(parents=True, exist_ok=True)
    (output / "reports").mkdir(parents=True, exist_ok=True)
    summary = {
        "source_dataset": args.source_dataset,
        "lookback_frames": args.lookback_frames,
        "episodes": len(rows),
        "official_full": sum(row["evaluation_tier"] == "official" for row in rows),
        "extended_backtracked": sum(row["evaluation_tier"] == "extended_backtracked" for row in rows),
        "rejected": len(rejected),
        "training_eligible": False,
    }
    manifest = {
        "version": 1,
        "name": args.output_dataset,
        "source_dataset": source_manifest["source_dataset"],
        "frequency_hz": source_manifest["frequency_hz"],
        "oracle_config": source_manifest.get("oracle_config", DEFAULT_ORACLE),
        "evaluation_only": True,
        "tasks": [{
            "task_id": 0,
            "calvin_task": "stack_block",
            "instruction": "stack the black-outlined source block on top of the cyan-outlined destination block",
        }],
        "splits": {"validation": {"summary": summary, "segments": rows}},
    }
    (output / "manifests/segments.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "reports/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "reports/rejected.json").write_text(json.dumps(rejected, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
