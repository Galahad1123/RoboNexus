"""Incremental rollout records and aggregate task metrics."""

import json
from collections import defaultdict
from pathlib import Path
import shutil


class ResultWriter:
    def __init__(self, output_dir, run_config, overwrite=False):
        self.output_dir = Path(output_dir)
        if overwrite and self.output_dir.exists():
            shutil.rmtree(str(self.output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.output_dir / "episodes.jsonl"
        if self.records_path.exists():
            raise FileExistsError(
                "Evaluation output already contains episodes: {}. "
                "Use a new --output-dir to avoid mixing runs.".format(self.records_path)
            )
        self.records = []
        (self.output_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n"
        )

    def append(self, record):
        self.records.append(record)
        with self.records_path.open("a") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")
        self.write_summary()

    def write_summary(self):
        by_task = defaultdict(list)
        for record in self.records:
            by_task[record["task"]].append(record)
        task_metrics = {}
        for task, records in sorted(by_task.items()):
            successes = sum(bool(record["success"]) for record in records)
            task_metrics[task] = {
                "episodes": len(records),
                "successes": successes,
                "success_rate": successes / len(records),
            }
        total = len(self.records)
        successes = sum(bool(record["success"]) for record in self.records)
        summary = {
            "episodes": total,
            "successes": successes,
            "success_rate": successes / total if total else 0.0,
            "macro_success_rate": (
                sum(item["success_rate"] for item in task_metrics.values()) / len(task_metrics)
                if task_metrics
                else 0.0
            ),
            "by_task": task_metrics,
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        return summary
