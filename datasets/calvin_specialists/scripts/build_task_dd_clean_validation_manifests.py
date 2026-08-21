#!/usr/bin/env python3
"""Build clean-instruction closed-loop evaluation manifests from task_D_D."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
GENERAL_SOURCE = WORKSPACE / "datasets/calvin_general_21/manifests/segments.json"
TARGET_SOURCES = {
    "calvin_abc_general_21_clean": (GENERAL_SOURCE,),
    "calvin_abc_lift_9_clean": (GENERAL_SOURCE,),
    "calvin_abc_device_6_clean": (GENERAL_SOURCE,),
    "calvin_abc_manipulation_13_clean": (
        WORKSPACE / "datasets/calvin_push_6/manifests/segments.json",
        WORKSPACE / "datasets/calvin_rotate_6_release/manifests/segments.json",
        WORKSPACE / "datasets/calvin_stack_pick_place_persistent/manifests/segments.json",
    ),
}


def main() -> None:
    source_dataset = "calvin/dataset/task_D_D"
    source_root = WORKSPACE / source_dataset / "validation"
    results = {}

    for name, source_paths in TARGET_SOURCES.items():
        dataset_root = WORKSPACE / "datasets" / name
        clean_manifest_path = dataset_root / "manifests/segments.json"
        clean = json.loads(clean_manifest_path.read_text())
        clean_tasks = {
            row["calvin_task"]: row["instruction"] for row in clean["tasks"]
        }
        task_order = sorted(clean_tasks)
        task_ids = {task: index for index, task in enumerate(task_order)}
        source_rows = []
        for source_path in source_paths:
            source = json.loads(source_path.read_text())
            if source["source_dataset"] != source_dataset:
                raise ValueError(f"Expected audited task_D_D source: {source_path}")
            source_rows.extend(source["splits"]["validation"]["segments"])
        rows = []
        for original in source_rows:
            task = original["calvin_task"]
            if task not in clean_tasks:
                continue
            row = dict(original)
            row["task_id"] = task_ids[task]
            row["instruction"] = clean_tasks[task]
            row["image_mode"] = "clean_rgb"
            if not row.get("oracle_consistent"):
                raise ValueError(
                    f"Source retained a non-oracle-consistent row: {row['segment_id']}"
                )
            first = source_root / f"episode_{int(row['start']):07d}.npz"
            last = source_root / f"episode_{int(row['end']):07d}.npz"
            if not first.is_file() or not last.is_file():
                raise FileNotFoundError(
                    f"Missing task_D_D source boundary for {row['segment_id']}: "
                    f"{first}, {last}"
                )
            rows.append(row)

        observed = {row["calvin_task"] for row in rows}
        if observed != set(task_order):
            raise ValueError(
                f"{name} task_D_D validation coverage mismatch: "
                f"missing={sorted(set(task_order) - observed)}"
            )
        if len(rows) != len({row["segment_id"] for row in rows}):
            raise ValueError(f"{name} task_D_D validation has duplicate segment IDs")

        counts = Counter(row["calvin_task"] for row in rows)
        summary = {
            "dataset": name,
            "source_dataset": source_dataset,
            "split": "validation",
            "episodes": len(rows),
            "frames": sum(int(row["num_frames"]) for row in rows),
            "by_task": dict(sorted(counts.items())),
            "oracle_consistent": True,
            "unique_segment_ids": True,
            "source_boundary_frames_exist": True,
            "instruction_mode": "clean_canonical",
        }
        manifest = {
            "version": 1,
            "name": f"{name}_task_D_D_validation",
            "source_dataset": source_dataset,
            "frequency_hz": 30,
            "image_mode": "clean_rgb",
            "evaluation_only": True,
            "derived_from": [str(path.relative_to(WORKSPACE)) for path in source_paths],
            "instruction_source": str(clean_manifest_path.relative_to(WORKSPACE)),
            "tasks": [
                {
                    "task_id": task_ids[task],
                    "calvin_task": task,
                    "instruction": clean_tasks[task],
                }
                for task in task_order
            ],
            "splits": {
                "validation": {
                    "source_split": "validation",
                    "summary": summary,
                    "segments": rows,
                }
            },
        }
        output = dataset_root / "manifests/task_D_D_validation.json"
        output.write_text(json.dumps(manifest, indent=2) + "\n")
        report = dataset_root / "reports/task_D_D_validation_summary.json"
        report.write_text(json.dumps(summary, indent=2) + "\n")
        results[name] = summary

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
