#!/usr/bin/env python3
"""Generate dependency-free offline HTML reports for RoboNexus evaluations."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote


TERMINAL_EVENTS = {"goal_succeeded": "succeeded", "goal_aborted": "aborted"}
VIDEO_PATTERN = re.compile(r"(?P<subtask>subtask_\d+)_attempt_(?P<attempt>\d+)\.mp4$")


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def load_events(path):
    events = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return events
    for number, line in enumerate(lines, 1):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"event": "malformed_log_line", "line_number": number})
    return events


def esc(value):
    return html.escape(str(value), quote=True)


def json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str)


def file_uri(path, report_dir):
    relative = path.relative_to(report_dir).as_posix()
    return "/".join(quote(part) for part in relative.split("/"))


def session_document(agent):
    return (agent or {}).get("session_after") or (agent or {}).get("session_before") or {}


def infer_status(runtime, agent, events):
    if runtime and runtime.get("session_status"):
        return runtime["session_status"]
    if agent and agent.get("effective_status"):
        return agent["effective_status"]
    for event in reversed(events):
        if event.get("event") in TERMINAL_EVENTS:
            return TERMINAL_EVENTS[event["event"]]
    return session_document(agent).get("session_status", "incomplete")


def qwen_timeline(agent):
    messages = (agent or {}).get("final_messages") or []
    timeline = []
    pending = None
    for message in messages:
        call = message.get("function_call") or {}
        if call.get("name"):
            arguments = call.get("arguments", "")
            try:
                arguments = json.loads(arguments)
            except (TypeError, json.JSONDecodeError):
                pass
            pending = {"tool": call["name"], "arguments": arguments, "result": None}
            timeline.append(pending)
        elif message.get("role") == "function":
            content = normalize_tool_result(message.get("content", ""))
            target = pending
            if target is None or target["result"] is not None:
                target = {"tool": message.get("name", "tool"), "arguments": {}, "result": None}
                timeline.append(target)
            target["result"] = content
    return timeline


def normalize_tool_result(content):
    """Extract JSON from plain or multimodal Qwen-Agent function content."""

    try:
        content = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                try:
                    return json.loads(item["text"])
                except json.JSONDecodeError:
                    continue
    return content


def build_model(directory):
    events = load_events(directory / "events.jsonl")
    runtime = load_json(directory / "runtime_result.json") or {}
    agent = load_json(directory / "qwen_agent_result.json")
    if agent is None:
        agent = load_json(directory / "qwen_agent_checkpoint.json") or {}
    session = session_document(agent)
    skills = list(session.get("allowed_skill_ids") or [])
    achieved = [e for e in events if e.get("event") == "skill_achieved"]
    if not skills:
        skills = [e.get("skill_id") for e in achieved if e.get("skill_id")]
    verifications = [
        e for e in events
        if e.get("event") == "verification_result" and e.get("episode_id")
    ]
    by_subtask = defaultdict(list)
    for verification in verifications:
        by_subtask[verification.get("episode_id")].append(verification)

    artifact_root = directory / "artifacts"
    files = sorted(path for path in artifact_root.rglob("*") if path.is_file()) if artifact_root.exists() else []
    videos = []
    for path in files:
        match = VIDEO_PATTERN.search(path.name)
        if match:
            videos.append({
                "path": path,
                "subtask": match.group("subtask"),
                "attempt": int(match.group("attempt")),
            })
    images = [path for path in files if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    timeline = qwen_timeline(agent)
    executions = {}
    for item in timeline:
        result = item.get("result")
        data = result.get("data", {}) if isinstance(result, dict) else {}
        if item.get("tool") == "execute_skill" and data.get("logical_subtask_id"):
            executions[(data["logical_subtask_id"], int(data.get("attempt", 0)))] = data
    retries = sum(max(0, len(items) - 1) for items in by_subtask.values())
    failures = sum(1 for item in verifications if item.get("status") != "success")
    status = infer_status(runtime, agent, events)
    terminal_reason = runtime.get("termination_reason")
    if not terminal_reason:
        terminal_reason = next(
            (e.get("reason") for e in reversed(events) if e.get("event") in TERMINAL_EVENTS), None
        )
    timestamps = [e.get("timestamp") for e in events if e.get("timestamp")]
    return {
        "directory": directory,
        "name": directory.name,
        "status": status,
        "reason": terminal_reason,
        "runtime": runtime,
        "agent": agent,
        "session": session,
        "events": events,
        "timeline": timeline,
        "executions": executions,
        "skills": skills,
        "achieved": achieved,
        "verifications": verifications,
        "by_subtask": by_subtask,
        "videos": videos,
        "images": images,
        "retries": retries,
        "failures": failures,
        "environment_steps": runtime.get("environment_steps", (session.get("budget") or {}).get("environment_steps_used")),
        "tool_calls": sum(1 for e in events if e.get("event") == "tool_call_started"),
        "recovered": bool(agent.get("recovered")),
        "started": timestamps[0] if timestamps else None,
        "finished": timestamps[-1] if timestamps else None,
    }


def badge(status):
    return '<span class="badge status-{}">{}</span>'.format(esc(status), esc(status))


def metric(label, value):
    return '<div class="metric"><span>{}</span><strong>{}</strong></div>'.format(esc(label), esc(value if value is not None else "—"))


def details(title, value, open_=False):
    return '<details{}><summary>{}</summary><pre>{}</pre></details>'.format(
        " open" if open_ else "", esc(title), esc(json_text(value))
    )


def render_media(model, report_dir):
    cards = []
    skills = model["skills"]
    for index, video in enumerate(model["videos"]):
        subtask_number = int(video["subtask"].split("_")[-1])
        skill = skills[subtask_number - 1] if subtask_number <= len(skills) else video["subtask"]
        checks = model["by_subtask"].get(video["subtask"], [])
        verification = next((v for v in checks if int(v.get("attempt", 0)) == video["attempt"]), None)
        execution = model["executions"].get((video["subtask"], video["attempt"]), {})
        state = (verification or {}).get("status", "unknown")
        nearby = [p for p in model["images"] if abs(int(p.name.split("_")[0]) - int(video["path"].name.split("_")[0])) <= 4]
        previews = "".join(
            '<a href="{0}"><img loading="lazy" src="{0}" alt="{1}"></a>'.format(
                file_uri(path, report_dir), esc(path.name)
            ) for path in nearby
        )
        cards.append(
            '<article class="attempt"><header><div><small>{}</small><h3>{}</h3></div>{}</header>'
            '<p class="attempt-meta">policy: <code>{}</code> · steps: <strong>{}</strong> · termination: {}</p>'
            '<video controls preload="metadata" src="{}"></video><div class="images">{}</div>{}{}</article>'.format(
                esc("{} · attempt {}".format(video["subtask"], video["attempt"])), esc(skill), badge(state),
                esc(execution.get("policy_id", "unknown")), esc(execution.get("steps", "—")), esc(execution.get("termination_reason", "—")),
                file_uri(video["path"], report_dir), previews,
                details("Execution result", execution) if execution else "",
                details("Oracle verification", verification) if verification else "",
            )
        )
    return "".join(cards) or '<p class="empty">No execution videos found.</p>'


def render_timeline(model):
    rows = []
    if model["timeline"]:
        for number, item in enumerate(model["timeline"], 1):
            result = item.get("result")
            state = result.get("status", "unknown") if isinstance(result, dict) else "unknown"
            rows.append(
                '<article class="timeline-item"><div class="rail">{}</div><div class="timeline-body">'
                '<header><code>{}</code>{}</header>{}{}</div></article>'.format(
                    number, esc(item["tool"]), badge(state),
                    details("Agent arguments", item.get("arguments", {})),
                    details("Runtime result", result) if result is not None else '<p class="empty">No result captured.</p>',
                )
            )
    else:
        for number, event in enumerate(model["events"], 1):
            rows.append(
                '<article class="timeline-item"><div class="rail">{}</div><div class="timeline-body">'
                '<header><code>{}</code><small>{}</small></header>{}</div></article>'.format(
                    number, esc(event.get("event", "event")), esc(event.get("timestamp", "")), details("Event", event)
                )
            )
    return "".join(rows)


CSS = """
:root{color-scheme:dark;--bg:#0b1020;--panel:#141b2d;--line:#29334d;--text:#edf2ff;--muted:#96a3bf;--accent:#7dd3fc;--ok:#34d399;--bad:#fb7185;--warn:#fbbf24}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#17213b,var(--bg) 42%);color:var(--text);font:15px/1.5 ui-sans-serif,system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:28px}h1,h2,h3{margin:.2rem 0}h1{font-size:clamp(25px,4vw,42px)}h2{margin-top:32px}.muted,small,.empty{color:var(--muted)}.top{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:22px 0}.metric,.panel,.attempt,.timeline-body{background:color-mix(in srgb,var(--panel) 94%,transparent);border:1px solid var(--line);border-radius:14px}.metric{padding:13px}.metric span{display:block;color:var(--muted);font-size:12px}.metric strong{font-size:22px}.panel{padding:18px;margin:14px 0}.skills{display:flex;flex-wrap:wrap;gap:8px}.skills code{padding:5px 9px;background:#222c45;border-radius:8px}.badge{display:inline-block;padding:3px 8px;border-radius:999px;font-size:12px;background:#33405f}.status-succeeded,.status-success{color:#06281e;background:var(--ok)}.status-failed,.status-aborted,.status-unavailable{color:#370810;background:var(--bad)}.status-active,.status-running,.status-incomplete,.status-unknown{color:#332500;background:var(--warn)}.attempts{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}.attempt{overflow:hidden}.attempt header{display:flex;justify-content:space-between;align-items:center;padding:14px 16px}.attempt-meta{margin:0;padding:0 16px 12px;color:var(--muted)}.attempt video{width:100%;max-height:430px;background:#000}.images{display:flex;gap:8px;padding:10px;overflow-x:auto}.images img{height:125px;border-radius:8px;border:1px solid var(--line)}details{border-top:1px solid var(--line);padding:9px 14px}summary{cursor:pointer;color:var(--accent)}pre{white-space:pre-wrap;word-break:break-word;max-height:500px;overflow:auto;background:#080c17;padding:12px;border-radius:8px}.timeline-item{display:grid;grid-template-columns:42px 1fr;margin:10px 0}.rail{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;background:#253252;color:var(--accent);margin-top:11px}.timeline-body header{display:flex;justify-content:space-between;padding:12px 14px}.toolbar{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}input,select{background:#0d1426;color:var(--text);border:1px solid var(--line);padding:9px 12px;border-radius:8px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:10px;border-bottom:1px solid var(--line)}a{color:var(--accent)}@media(max-width:650px){main{padding:15px}.attempts{grid-template-columns:1fr}.top{display:block}table{font-size:12px}.hide-mobile{display:none}}
"""


def page(title, body, script=""):
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{}</title><style>{}</style></head><body><main>{}</main><script>{}</script></body></html>'.format(esc(title), CSS, body, script)


def render_sequence(model, destination):
    report_dir = destination.parent
    session = model["session"]
    body = '<div class="top"><div><p class="muted">RoboNexus offline evaluation report</p><h1>{}</h1><p>{}</p></div>{}</div>'.format(
        esc(model["name"]), esc(session.get("goal_instruction", "Goal instruction unavailable")), badge(model["status"])
    )
    body += '<section class="metrics">{}</section>'.format("".join([
        metric("Completed skills", "{}/{}".format(len(model["achieved"]), len(model["skills"]))),
        metric("Environment steps", model["environment_steps"]), metric("Tool calls", model["tool_calls"]),
        metric("Retries", model["retries"]), metric("Failed attempts", model["failures"]),
        metric("Recovered", "yes" if model["recovered"] else "no"),
    ]))
    body += '<section class="panel"><h2>Goal</h2><p><strong>Termination:</strong> {}</p><div class="skills">{}</div><p class="muted">{} → {}</p></section>'.format(
        esc(model["reason"] or "—"), "".join('<code>{}</code>'.format(esc(skill)) for skill in model["skills"]),
        esc(model["started"] or "unknown"), esc(model["finished"] or "unknown")
    )
    body += '<h2>DiT executions</h2><section class="attempts">{}</section>'.format(render_media(model, report_dir))
    body += '<h2>Agent decisions and tool calls</h2><section>{}</section>'.format(render_timeline(model))
    body += '<section class="panel"><h2>Raw summaries</h2>{}{}</section>'.format(details("Runtime result", model["runtime"]), details("Agent result metadata", {k:v for k,v in model["agent"].items() if k != "final_messages"}))
    destination.write_text(page(model["name"], body), encoding="utf-8")


def discover_sequences(root):
    candidates = set()
    for name in ("events.jsonl", "runtime_result.json", "qwen_agent_result.json", "qwen_agent_checkpoint.json"):
        candidates.update(path.parent for path in root.rglob(name))
    return sorted(candidates)


def render_index(models, root, destination):
    counts = Counter(model["status"] for model in models)
    successes = counts["succeeded"]
    total = len(models)
    retries = sum(model["retries"] for model in models)
    completed_skills = sum(len(model["achieved"]) for model in models)
    requested_skills = sum(len(model["skills"]) for model in models)
    rows = []
    for model in models:
        report = model["directory"] / "report.html"
        relative = file_uri(report, destination.parent)
        search = " ".join([model["name"], model["status"], model["reason"] or "", *model["skills"]]).lower()
        rows.append('<tr data-status="{}" data-search="{}"><td><a href="{}">{}</a></td><td>{}</td><td>{}/{}</td><td>{}</td><td>{}</td><td class="hide-mobile">{}</td></tr>'.format(
            esc(model["status"]), esc(search), relative, esc(model["name"]), badge(model["status"]),
            len(model["achieved"]), len(model["skills"]), model["retries"], esc(model["environment_steps"] if model["environment_steps"] is not None else "—"), esc(model["reason"] or "—")
        ))
    body = '<p class="muted">RoboNexus offline evaluation dashboard</p><h1>{}</h1>'.format(esc(root.name))
    body += '<section class="metrics">{}</section>'.format("".join([
        metric("Sequences", total), metric("Success rate", "{:.1f}%".format(100 * successes / total if total else 0)),
        metric("Succeeded", successes), metric("Aborted", counts["aborted"]), metric("Total retries", retries),
        metric("Completed skills", "{}/{}".format(completed_skills, requested_skills)),
    ]))
    body += '<div class="toolbar"><input id="search" placeholder="Search goal, skill, reason…"><select id="status"><option value="">All statuses</option>{}</select></div>'.format(
        "".join('<option value="{}">{} ({})</option>'.format(esc(key), esc(key), value) for key,value in sorted(counts.items()))
    )
    body += '<section class="panel"><table><thead><tr><th>Sequence</th><th>Status</th><th>Skills</th><th>Retries</th><th>Steps</th><th class="hide-mobile">Reason</th></tr></thead><tbody id="runs">{}</tbody></table></section>'.format("".join(rows))
    script = """const q=document.querySelector('#search'),s=document.querySelector('#status');function f(){for(const r of document.querySelectorAll('#runs tr')){r.hidden=!!s.value&&r.dataset.status!==s.value||!r.dataset.search.includes(q.value.toLowerCase())}}q.addEventListener('input',f);s.addEventListener('change',f);"""
    destination.write_text(page(root.name, body, script), encoding="utf-8")


def generate(root, recursive=False):
    directories = discover_sequences(root) if recursive else [root]
    if not directories:
        raise SystemExit("no evaluation outputs found under {}".format(root))
    models = []
    for directory in directories:
        model = build_model(directory)
        render_sequence(model, directory / "report.html")
        models.append(model)
    if recursive:
        render_index(models, root, root / "index.html")
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--recursive", action="store_true", help="generate per-sequence reports and a batch index")
    args = parser.parse_args()
    models = generate(args.output_dir.resolve(), args.recursive)
    target = args.output_dir / ("index.html" if args.recursive else "report.html")
    print("generated {} report(s): {}".format(len(models), target.resolve()))


if __name__ == "__main__":
    main()
