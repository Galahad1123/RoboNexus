#!/usr/bin/env python3
"""Run official CALVIN sequences while keeping one Qwen model resident."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError


WORKSPACE = Path(__file__).resolve().parents[2]
AGENT_ROOT = WORKSPACE / "RoboNexus-Agent"
for path in (AGENT_ROOT, WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adapters.qwen_agent import build_qwen_llm, fetch_session, run_single_skill_agent  # noqa: E402
from core.logging import (  # noqa: E402
    atomic_write_json,
    read_json,
    recover_agent_result,
    terminal_status_from_events,
)
from core.skills import load_skill_registry  # noqa: E402


SUITES = {
    "general": "datasets/calvin_general_21/configs/evaluation.yaml",
    "push": "datasets/calvin_push_6/configs/evaluation.yaml",
    "rotate": "datasets/calvin_rotate_6_release/configs/evaluation.yaml",
    "stack": "datasets/calvin_full_stack_extended_eval/configs/evaluation.yaml",
}


def wait_for_runtime(url, process, timeout, expected_skills):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("CALVIN Runtime exited with code {} before readiness".format(process.returncode))
        try:
            session = fetch_session(url, timeout=2.0)
            callable_ids = session.get("callable_skill_ids", session.get("allowed_skill_ids"))
            if callable_ids == list(expected_skills):
                return session
        except (URLError, ConnectionError, TimeoutError):
            time.sleep(0.5)
    raise TimeoutError("CALVIN Runtime did not become ready within {} seconds".format(timeout))


def stop_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def sequence_name(item):
    return "sequence_{:04d}_{}".format(item["index"], "__".join(item["skills"]))


def is_terminal(status):
    return status not in {None, "pending", "active", "waiting_for_user"}


def sequence_status(directory):
    runtime = read_json(directory / "runtime_result.json") or {}
    runtime_status = runtime.get("session_status")
    event_status = terminal_status_from_events(directory / "events.jsonl")
    agent = read_json(directory / "qwen_agent_result.json") or {}
    if is_terminal(runtime_status):
        return runtime_status
    return event_status or agent.get("effective_status") or runtime_status or "pending"


def batch_summary(root, manifest, started_at):
    rows = []
    for item in manifest["sequences"]:
        directory = root / sequence_name(item)
        runtime = read_json(directory / "runtime_result.json") or {}
        status = sequence_status(directory)
        terminal_event = None
        for event in reversed(_read_events(directory / "events.jsonl")):
            if event.get("event") in {"goal_succeeded", "goal_aborted"}:
                terminal_event = event
                break
        rows.append(
            {
                "index": item["index"],
                "skills": item["skills"],
                "status": status,
                "termination_reason": runtime.get("termination_reason") or (terminal_event or {}).get("reason"),
                "environment_steps": runtime.get("environment_steps"),
            }
        )
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {"started_at": started_at, "updated_at": time.time(), "counts": counts, "sequences": rows}


def _read_events(path):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=WORKSPACE / "weights/Qwen3-VL-4B-Instruct")
    parser.add_argument("--runtime-port", type=int, default=8121)
    parser.add_argument("--runtime-env", default="robonexus-calvin")
    parser.add_argument("--runtime-ready-timeout", type=float, default=180.0)
    parser.add_argument("--environment-step-limit", type=int, default=2200)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-agent-rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--approach", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    sequences = manifest["sequences"][: args.limit]
    manifest = dict(manifest)
    manifest["sequences"] = sequences
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "sequence_manifest.json", manifest)
    registry = load_skill_registry(AGENT_ROOT / "configs/calvin/skills_d34.yaml")
    llm = build_qwen_llm(args.model_path, args.max_new_tokens)
    started_at = time.time()
    runtime_url = "http://127.0.0.1:{}".format(args.runtime_port)

    for position, item in enumerate(sequences, 1):
        output = args.output_dir / sequence_name(item)
        if is_terminal(sequence_status(output)):
            continue
        output.mkdir(parents=True, exist_ok=True)
        first = registry.get(item["skills"][0])
        goal = "Complete these robot goal steps in the exact numbered order:\n" + "\n".join(
            "{}. {}".format(number, registry.get(skill_id).planner_instruction)
            for number, skill_id in enumerate(item["skills"], 1)
        )
        command = [
            "conda", "run", "-n", args.runtime_env, "env", "PYTHONPATH=RoboNexus-Agent:.",
            "python", "scripts/agent/serve_calvin_tool_session.py",
            "--port", str(args.runtime_port), "--task", item["skills"][0],
            "--goal", goal, "--suite", str(WORKSPACE / SUITES[first.policy_group]),
            "--official-initial-condition", json.dumps(item["initial_condition"], separators=(",", ":")),
            "--environment-step-limit", str(args.environment_step_limit),
            "--terminal-grace-seconds", "2", "--output-dir", str(output),
        ]
        for skill in item["skills"]:
            command.extend(["--skill", skill])
        if not args.approach:
            command.append("--disable-approach")
        with (output / "runtime.log").open("a", encoding="utf-8") as runtime_log:
            process = subprocess.Popen(
                command,
                cwd=WORKSPACE,
                stdout=runtime_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                wait_for_runtime(
                    runtime_url, process, args.runtime_ready_timeout, registry.skill_ids()
                )
                run_single_skill_agent(
                    runtime_url, args.model_path, seed=args.seed + item["index"],
                    max_new_tokens=args.max_new_tokens, output=output / "qwen_agent_result.json",
                    expected_skill_ids=registry.skill_ids(), llm=llm,
                    max_agent_rounds=args.max_agent_rounds,
                )
                process.wait(timeout=20)
            except BaseException as error:
                atomic_write_json(output / "batch_error.json", {"type": type(error).__name__, "message": str(error)})
                stop_process(process)
                try:
                    recover_agent_result(output)
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
            finally:
                stop_process(process)
        atomic_write_json(args.output_dir / "batch_result.json", batch_summary(args.output_dir, manifest, started_at))
        subprocess.run([sys.executable, "scripts/agent/generate_eval_report.py", str(args.output_dir), "--recursive"], cwd=WORKSPACE, check=False)
        print("[{}/{}] {}".format(position, len(sequences), sequence_name(item)), flush=True)

    summary = batch_summary(args.output_dir, manifest, started_at)
    atomic_write_json(args.output_dir / "batch_result.json", summary)
    subprocess.run([sys.executable, "scripts/agent/generate_eval_report.py", str(args.output_dir), "--recursive"], cwd=WORKSPACE, check=True)
    print(json.dumps(summary["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
