#!/usr/bin/env python3
"""Derive audited clean-RGB Lift-9 and Device-6 manifests from General-21."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[2]
DEFAULT_CONFIG = (
    WORKSPACE
    / "datasets/calvin_specialists/configs/calvin_abc_lift_device_clean.json"
)


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else WORKSPACE / path


def summarize(name: str, split: str, rows: list[dict]) -> dict:
    by_task = Counter(row["calvin_task"] for row in rows)
    by_environment = Counter(row["environment"] for row in rows)
    task_environment = Counter(
        (row["calvin_task"], row["environment"]) for row in rows
    )
    return {
        "dataset": name,
        "split": split,
        "episodes": len(rows),
        "frames": sum(int(row["num_frames"]) for row in rows),
        "by_task": dict(sorted(by_task.items())),
        "by_environment": dict(sorted(by_environment.items())),
        "by_task_environment": {
            task: {
                environment: task_environment[(task, environment)]
                for environment in ("A", "B", "C", "D")
                if task_environment[(task, environment)]
            }
            for task in sorted(by_task)
        },
    }


def filter_report(path: Path, tasks: set[str]) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        row
        for row in json.loads(path.read_text())
        if row.get("calvin_task", row.get("task")) in tasks
    ]


def build_dataset(name: str, cfg: dict, parent_path: Path, parent: dict) -> dict:
    task_names = list(cfg["tasks"])
    tasks = set(task_names)
    if len(tasks) != len(task_names):
        raise ValueError(f"{name} contains duplicate task names")
    parent_tasks = {row["calvin_task"]: row for row in parent["tasks"]}
    unknown = sorted(tasks - set(parent_tasks))
    if unknown:
        raise ValueError(f"{name} contains tasks absent from parent manifest: {unknown}")

    task_order = sorted(tasks)
    task_ids = {task: index for index, task in enumerate(task_order)}
    source_root = resolve(parent["source_dataset"])
    split_documents = {}
    summaries = {}
    audit_checks = {
        "parent_manifest": str(parent_path.relative_to(WORKSPACE)),
        "parent_image_mode": parent.get("image_mode"),
        "tasks_requested": task_order,
        "checks": {},
    }
    for split, split_doc in parent["splits"].items():
        source_split = split_doc.get("source_split", split)
        source_dir = source_root / source_split
        rows = []
        missing_frames = []
        for parent_row in split_doc["segments"]:
            if parent_row["calvin_task"] not in tasks:
                continue
            row = dict(parent_row)
            row["task_id"] = task_ids[row["calvin_task"]]
            if not row.get("oracle_consistent"):
                raise ValueError(
                    f"parent retained a non-oracle-consistent row: {row['segment_id']}"
                )
            first = source_dir / f"episode_{int(row['start']):07d}.npz"
            last = source_dir / f"episode_{int(row['end']):07d}.npz"
            if not first.is_file() or not last.is_file():
                missing_frames.append(
                    {"segment_id": row["segment_id"], "first": str(first), "last": str(last)}
                )
            rows.append(row)
        observed = {row["calvin_task"] for row in rows}
        missing_tasks = sorted(tasks - observed)
        if missing_tasks or missing_frames:
            raise ValueError(
                f"{name}/{split} audit failed: missing_tasks={missing_tasks}, "
                f"missing_frames={missing_frames[:3]}"
            )
        if len(rows) != len({row["segment_id"] for row in rows}):
            raise ValueError(f"{name}/{split} contains duplicate segment IDs")
        summary = summarize(name, split, rows)
        summaries[split] = summary
        split_documents[split] = {
            "source_split": source_split,
            "summary": summary,
            "segments": rows,
        }
        audit_checks["checks"][split] = {
            "tasks_present": sorted(observed),
            "episodes": len(rows),
            "oracle_consistent": all(row["oracle_consistent"] for row in rows),
            "unique_segment_ids": True,
            "source_boundary_frames_exist": True,
        }

    output = WORKSPACE / "datasets" / name
    (output / "manifests").mkdir(parents=True, exist_ok=True)
    (output / "reports").mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "name": name,
        "source_dataset": parent["source_dataset"],
        "frequency_hz": parent["frequency_hz"],
        "image_mode": "clean_rgb",
        "derived_from": str(parent_path.relative_to(WORKSPACE)),
        "abc_validation_ratio": parent.get("abc_validation_ratio"),
        "split_seed": parent.get("split_seed"),
        "tasks": [
            {
                "task_id": task_ids[task],
                "calvin_task": task,
                "instruction": parent_tasks[task]["instruction"],
            }
            for task in task_order
        ],
        "splits": split_documents,
    }
    (output / "manifests/segments.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (output / "reports/summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    (output / "reports/derivation_audit.json").write_text(
        json.dumps(audit_checks, indent=2) + "\n"
    )

    parent_reports = parent_path.parents[1] / "reports"
    for filename in (
        "abc_oracle_audit.json",
        "d_oracle_audit.json",
        "abc_overlap_clusters.json",
        "d_overlap_clusters.json",
    ):
        filtered = filter_report(parent_reports / filename, tasks)
        (output / "reports" / filename).write_text(
            json.dumps(filtered, indent=2) + "\n"
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", action="append")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    selected = args.dataset or sorted(config["datasets"])
    unknown = sorted(set(selected) - set(config["datasets"]))
    if unknown:
        parser.error(f"unknown datasets: {unknown}")
    parent_path = resolve(config["parent_manifest"]).resolve()
    parent = json.loads(parent_path.read_text())
    if parent.get("image_mode") != "clean_rgb":
        raise ValueError("parent manifest must be audited for clean RGB")
    results = {
        name: build_dataset(
            name, config["datasets"][name], parent_path, parent
        )
        for name in selected
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
