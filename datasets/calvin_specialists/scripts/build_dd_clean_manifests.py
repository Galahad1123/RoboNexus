#!/usr/bin/env python3
"""Build four clean-RGB CALVIN-D manifests from existing audited manifests."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[2]
DEFAULT_CONFIG = WORKSPACE / "datasets/calvin_specialists/configs/calvin_dd_clean.json"
FORBIDDEN_PROMPT_TOKENS = ("outline", "outlined", "cyan")


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else WORKSPACE / path


def summarize(name: str, split: str, rows: list[dict]) -> dict:
    by_task = Counter(row["calvin_task"] for row in rows)
    by_variant = Counter(row.get("episode_variant", "standard") for row in rows)
    return {
        "dataset": name,
        "split": split,
        "episodes": len(rows),
        "frames": sum(int(row["num_frames"]) for row in rows),
        "by_task": dict(sorted(by_task.items())),
        "by_episode_variant": dict(sorted(by_variant.items())),
    }


def load_parent(name: str) -> tuple[Path, dict]:
    path = WORKSPACE / "datasets" / name / "manifests/segments.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, json.loads(path.read_text())


def build_dataset(name: str, cfg: dict, global_cfg: dict) -> dict:
    instructions = cfg["tasks"]
    tasks = set(instructions)
    if len(tasks) != len(instructions):
        raise ValueError(f"{name}: duplicate tasks")
    bad_prompts = {
        task: prompt
        for task, prompt in instructions.items()
        if any(token in prompt.lower() for token in FORBIDDEN_PROMPT_TOKENS)
    }
    if bad_prompts:
        raise ValueError(f"{name}: non-clean prompts: {bad_prompts}")

    parents = [load_parent(parent) for parent in cfg["sources"]]
    parent_tasks = {
        row["calvin_task"]
        for _, parent in parents
        for row in parent["tasks"]
    }
    missing = sorted(tasks - parent_tasks)
    if missing:
        raise ValueError(f"{name}: tasks absent from source manifests: {missing}")

    task_order = sorted(tasks)
    task_ids = {task: index for index, task in enumerate(task_order)}
    source_root = resolve(global_cfg["source_dataset"])
    split_documents = {}
    summaries = {}
    checks = {}
    for split in ("training", "validation"):
        rows = []
        for parent_path, parent in parents:
            source_split = parent["splits"][split].get("source_split", split)
            if source_split != split:
                raise ValueError(
                    f"{parent_path}: expected source split {split}, got {source_split}"
                )
            for source_row in parent["splits"][split]["segments"]:
                task = source_row["calvin_task"]
                if task not in tasks:
                    continue
                if not source_row.get("oracle_consistent"):
                    raise ValueError(
                        f"{name}: non-oracle row {source_row['segment_id']}"
                    )
                row = dict(source_row)
                row["task_id"] = task_ids[task]
                row["instruction"] = instructions[task]
                row["audit_source_manifest"] = str(parent_path.relative_to(WORKSPACE))
                rows.append(row)

        rows.sort(key=lambda row: (row["task_id"], row["start"], row["end"]))
        observed = {row["calvin_task"] for row in rows}
        missing_tasks = sorted(tasks - observed)
        segment_keys = {
            (row["audit_source_manifest"], row["segment_id"]) for row in rows
        }
        if missing_tasks:
            raise ValueError(f"{name}/{split}: missing tasks: {missing_tasks}")
        if len(segment_keys) != len(rows):
            raise ValueError(f"{name}/{split}: duplicate source segment keys")

        missing_frames = []
        for row in rows:
            for frame in (int(row["start"]), int(row["end"])):
                path = source_root / split / f"episode_{frame:07d}.npz"
                if not path.is_file():
                    missing_frames.append(str(path))
        if missing_frames:
            raise ValueError(
                f"{name}/{split}: missing boundary frames: {missing_frames[:3]}"
            )

        summary = summarize(name, split, rows)
        summaries[split] = summary
        split_documents[split] = {
            "source_split": split,
            "summary": summary,
            "segments": rows,
        }
        checks[split] = {
            "tasks_present": sorted(observed),
            "oracle_consistent": True,
            "unique_source_segment_keys": True,
            "source_boundary_frames_exist": True,
            "clean_instructions": True,
        }

    output = WORKSPACE / "datasets" / name
    (output / "manifests").mkdir(parents=True, exist_ok=True)
    (output / "reports").mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "name": name,
        "source_dataset": global_cfg["source_dataset"],
        "frequency_hz": global_cfg["frequency_hz"],
        "image_mode": "clean_rgb",
        "derived_from": [str(path.relative_to(WORKSPACE)) for path, _ in parents],
        "tasks": [
            {
                "task_id": task_ids[task],
                "calvin_task": task,
                "instruction": instructions[task],
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
        json.dumps(
            {
                "dataset": name,
                "image_mode": "clean_rgb",
                "sources": manifest["derived_from"],
                "checks": checks,
            },
            indent=2,
        ) + "\n"
    )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", action="append")
    args = parser.parse_args()
    config = json.loads(resolve(args.config).read_text())
    selected = args.dataset or sorted(config["datasets"])
    unknown = sorted(set(selected) - set(config["datasets"]))
    if unknown:
        parser.error(f"unknown datasets: {unknown}")

    configured_tasks = [
        task
        for dataset in config["datasets"].values()
        for task in dataset["tasks"]
    ]
    if len(configured_tasks) != 34 or len(set(configured_tasks)) != 34:
        raise ValueError("The four clean datasets must partition exactly 34 tasks")

    results = {
        name: build_dataset(name, config["datasets"][name], config)
        for name in selected
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
