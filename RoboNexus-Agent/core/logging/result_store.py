"""Crash-tolerant JSON result persistence and Agent-run recovery."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Durably replace *path* with one complete JSON document."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(destination.name), suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_json(path: str | Path) -> dict[str, Any] | None:
    source = Path(path)
    if not source.is_file():
        return None
    return json.loads(source.read_text(encoding="utf-8"))


def terminal_status_from_events(path: str | Path) -> str | None:
    """Read the last valid terminal event, tolerating a truncated final line."""

    source = Path(path)
    if not source.is_file():
        return None
    terminal_status = None
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "goal_succeeded":
                terminal_status = "succeeded"
            elif event.get("event") == "goal_aborted":
                terminal_status = "aborted"
    return terminal_status


def recover_agent_result(output_dir: str | Path) -> dict[str, Any]:
    """Reconcile an interrupted Qwen run with the authoritative robot runtime."""

    directory = Path(output_dir)
    result_path = directory / "qwen_agent_result.json"
    checkpoint_path = directory / "qwen_agent_checkpoint.json"
    runtime_path = directory / "runtime_result.json"
    existing = read_json(result_path)
    checkpoint = read_json(checkpoint_path)
    runtime = read_json(runtime_path)
    if existing is None and checkpoint is None and runtime is None:
        raise FileNotFoundError("no Agent or runtime result found in {}".format(directory))

    report = dict(existing or checkpoint or {})
    events_path = directory / "events.jsonl"
    runtime_status = (runtime or {}).get("session_status")
    event_status = terminal_status_from_events(events_path)
    agent_status = (report.get("session_after") or {}).get("session_status")
    effective_status = runtime_status or event_status or agent_status or "unknown"
    report.update(
        {
            "run_state": "completed" if effective_status in {"succeeded", "aborted"} else "interrupted",
            "effective_status": effective_status,
            "runtime_result": runtime,
            "recovered": True,
            "recovery_sources": {
                "agent_result": result_path.is_file(),
                "agent_checkpoint": checkpoint_path.is_file(),
                "runtime_result": runtime_path.is_file(),
                "events": events_path.is_file(),
            },
        }
    )
    atomic_write_json(result_path, report)
    return report
