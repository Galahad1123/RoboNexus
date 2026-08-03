"""Incremental rollout records and aggregate task metrics."""

import argparse
import csv
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
        write_video_index(self.records, self.output_dir / "video_index.csv")

    def write_summary(self):
        by_task = defaultdict(list)
        by_variant = defaultdict(list)
        for record in self.records:
            by_task[record["task"]].append(record)
            if record.get("episode_variant"):
                by_variant[record["episode_variant"]].append(record)
        task_metrics = {}
        for task, records in sorted(by_task.items()):
            successes = sum(bool(record["success"]) for record in records)
            first_attempt_failures = sum(
                bool(record.get("attempts"))
                and not bool(record["attempts"][0].get("success"))
                for record in records
            )
            recovered_after_failure = sum(
                bool(record.get("success"))
                and int(record.get("attempt_count", 1)) > 1
                for record in records
            )
            success_at = {}
            for attempt_limit in (1, 2, 3):
                recovered = sum(
                    bool(record.get("success"))
                    and int(record.get("attempt_count", 1)) <= attempt_limit
                    for record in records
                )
                success_at[str(attempt_limit)] = recovered / len(records)
            task_metrics[task] = {
                "episodes": len(records),
                "successes": successes,
                "success_rate": successes / len(records),
                "success_at_attempt": success_at,
                "mean_attempt_count": sum(
                    int(record.get("attempt_count", 1)) for record in records
                )
                / len(records),
                "recovery_failures": sum(
                    record.get("termination_reason") in {
                        "recovery_failed",
                        "recovery_exception",
                    }
                    for record in records
                ),
                "conditional_recovery_rate": (
                    recovered_after_failure / first_attempt_failures
                    if first_attempt_failures
                    else 0.0
                ),
            }
        total = len(self.records)
        successes = sum(bool(record["success"]) for record in self.records)
        first_attempt_failures = sum(
            bool(record.get("attempts"))
            and not bool(record["attempts"][0].get("success"))
            for record in self.records
        )
        recovered_after_failure = sum(
            bool(record.get("success"))
            and int(record.get("attempt_count", 1)) > 1
            for record in self.records
        )
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
            "by_episode_variant": {
                variant: {
                    "episodes": len(records),
                    "successes": sum(bool(record["success"]) for record in records),
                    "success_rate": sum(bool(record["success"]) for record in records)
                    / len(records),
                }
                for variant, records in sorted(by_variant.items())
            },
            "success_at_attempt": {
                str(attempt_limit): (
                    sum(
                        bool(record.get("success"))
                        and int(record.get("attempt_count", 1)) <= attempt_limit
                        for record in self.records
                    )
                    / total
                    if total
                    else 0.0
                )
                for attempt_limit in (1, 2, 3)
            },
            "policy_steps": sum(
                int(record.get("policy_steps", record.get("steps_executed", 0)))
                for record in self.records
            ),
            "recovery_steps": sum(
                int(record.get("recovery_steps", 0)) for record in self.records
            ),
            "conditional_recovery_rate": (
                recovered_after_failure / first_attempt_failures
                if first_attempt_failures
                else 0.0
            ),
            "retry_disabled_episodes": sum(
                not bool(record.get("retry_enabled", True))
                for record in self.records
            ),
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        return summary


def write_video_index(records, output_path):
    """Write a human-readable mapping from videos to task semantics."""
    fields = [
        "task",
        "episode_variant",
        "instruction",
        "episode_id",
        "success",
        "success_step",
        "termination_reason",
        "video",
    ]
    with Path(output_path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            if not record.get("video"):
                continue
            writer.writerow({field: record.get(field) for field in fields})


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild video_index.csv for an existing evaluation run."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    records_path = args.output_dir / "episodes.jsonl"
    if not records_path.is_file():
        raise FileNotFoundError("Missing evaluation records: {}".format(records_path))
    records = [
        json.loads(line)
        for line in records_path.read_text().splitlines()
        if line.strip()
    ]
    output_path = args.output_dir / "video_index.csv"
    write_video_index(records, output_path)
    print("Wrote {} entries to {}".format(len(records), output_path))


if __name__ == "__main__":
    main()
