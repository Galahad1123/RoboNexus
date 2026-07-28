#!/usr/bin/env python3
"""Summarize canonical and natural-language task labels in a CALVIN dataset."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "calvin" / "dataset" / "task_D_D"


def analyze_split(dataset: Path, split: str, frequency: float) -> Dict[str, Any]:
    path = dataset / split / "lang_annotations" / "auto_lang_ann.npy"
    data = np.load(path, allow_pickle=True).item()
    tasks = list(data["language"]["task"])
    annotations = list(data["language"]["ann"])
    intervals = np.asarray(data["info"]["indx"], dtype=np.int64)
    if not (len(tasks) == len(annotations) == len(intervals)):
        raise ValueError(f"inconsistent annotation lengths in {path}")

    grouped = defaultdict(lambda: {"segments": 0, "frames": 0, "lengths": [], "utterances": Counter()})
    for task, annotation, (start, end) in zip(tasks, annotations, intervals):
        length = int(end - start + 1)
        row = grouped[str(task)]
        row["segments"] += 1
        row["frames"] += length
        row["lengths"].append(length)
        row["utterances"][str(annotation)] += 1

    result = {}
    for task in sorted(grouped):
        row = grouped[task]
        lengths = np.asarray(row.pop("lengths"))
        utterances = row.pop("utterances")
        result[task] = {
            **row,
            "hours": round(row["frames"] / frequency / 3600, 4),
            "length_frames": {
                "min": int(lengths.min()),
                "median": float(np.median(lengths)),
                "mean": round(float(lengths.mean()), 2),
                "p95": float(np.percentile(lengths, 95)),
                "max": int(lengths.max()),
            },
            "unique_utterances": len(utterances),
            "top_utterances": [
                {"text": text, "segments": count} for text, count in utterances.most_common(3)
            ],
        }

    return {
        "annotation_file": str(path),
        "frequency_hz": frequency,
        "segments": len(tasks),
        "tasks": len(result),
        "labeled_frames": int(sum(row["frames"] for row in result.values())),
        "task_stats": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "dataset": str(args.dataset.resolve()),
        "splits": {
            split: analyze_split(args.dataset.resolve(), split, args.frequency)
            for split in ("training", "validation")
        },
    }
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
