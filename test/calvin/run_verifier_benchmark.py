#!/usr/bin/env python3
"""Run Qwen verifier on an Oracle-labelled CALVIN-D manifest and render HTML."""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from subtask_verifier import FramePair, QwenVerifierBackend, run_verification


def qwen_success(result: Dict[str, Any]) -> bool:
    return result["decision"]["status"] == "success"


def evaluate(
    manifest_path: Path,
    model_path: Path,
    output: Path,
    max_new_tokens: int,
    use_goal_reference: bool,
) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    backend = QwenVerifierBackend(model_path, max_new_tokens)
    rows = []
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if bool(previous.get("use_goal_reference")) != bool(use_goal_reference):
            raise ValueError("existing output uses a different goal-reference mode")
        rows = list(previous.get("rows", []))
    completed_ids = {row["case_id"] for row in rows}
    for case in manifest["cases"]:
        if case["case_id"] in completed_ids:
            continue
        static_path = manifest_path.parent / case["static_image"]
        is_close = case["task"] == "close_drawer"
        is_light = case["task"] in {
            "turn_on_lightbulb", "turn_off_lightbulb", "turn_on_led", "turn_off_led"
        }
        static_only = is_close or is_light
        wrist_path = manifest_path.parent / case["wrist_image"]
        frame = FramePair(
            Image.open(static_path).convert("RGB"),
            None if static_only else Image.open(wrist_path).convert("RGB"),
            "current",
        )
        initial_state = FramePair(
            Image.open(manifest_path.parent / case["initial_static_image"]).convert("RGB"),
            None
            if static_only
            else Image.open(manifest_path.parent / case["initial_wrist_image"]).convert("RGB"),
            "initial",
        )
        history = [
            FramePair(
                Image.open(manifest_path.parent / item["static_image"]).convert("RGB"),
                None
                if static_only
                else Image.open(manifest_path.parent / item["wrist_image"]).convert("RGB"),
                "recent_{}".format(index),
            )
            for index, item in enumerate(case.get("history", []), 1)
        ]
        reference_path = case.get("goal_reference_static")
        goal_reference = None
        goal_reference_wrist = None
        if use_goal_reference:
            if not reference_path:
                raise ValueError("case {} has no goal reference".format(case["case_id"]))
            goal_reference = Image.open(manifest_path.parent / reference_path).convert("RGB")
            wrist_reference_path = case.get("goal_reference_wrist")
            if wrist_reference_path:
                goal_reference_wrist = Image.open(
                    manifest_path.parent / wrist_reference_path
                ).convert("RGB")
        is_lift = case["task"].startswith("lift_")
        observation_context = dict(case.get("observation_context", {}))
        if is_lift:
            observation_context["previous_frame_ids"] = []
            observation_context["input_mode"] = "current_static_current_wrist_goal_wrist_only"
        elif is_close:
            close_current_path = case.get("close_initial_current_static")
            if not close_current_path:
                raise ValueError("close_drawer case is missing initial/current comparison")
            frame = FramePair(
                Image.open(manifest_path.parent / close_current_path).convert("RGB"),
                None,
                "initial_top_current_bottom_comparison",
            )
            history = []
            initial_state = None
            observation_context["input_mode"] = (
                "close_initial_current_and_goal_composites_static_only"
            )
        elif case["task"] in {"turn_on_lightbulb", "turn_on_led", "turn_off_lightbulb"}:
            history = []
            initial_state = None
            observation_context["previous_frame_ids"] = []
            observation_context["input_mode"] = "current_task_crop_static_only"
        elif is_light:
            history = []
            initial_state = None
            observation_context["previous_frame_ids"] = []
            observation_context["input_mode"] = "current_crop_and_goal_crop_static_only"
        elif case["task"] == "move_slider_right":
            history = []
            initial_state = None
            frame = FramePair(frame.static, None, "current")
            observation_context["previous_frame_ids"] = []
            observation_context["input_mode"] = "current_static_goal_static_only"
        elif case["task"] == "place_in_slider":
            history = []
            initial_state = None
            observation_context["previous_frame_ids"] = []
            observation_context["input_mode"] = (
                "current_static_current_wrist_goal_wrist_only"
            )
        elif case["task"] == "unstack_block":
            history = []
            initial_state = None
            observation_context["previous_frame_ids"] = []
            observation_context["input_mode"] = "current_static_current_wrist_only"
        elif case["task"].startswith("rotate_"):
            history = []
            initial_state = None
            frame = FramePair(None, frame.wrist, "current")
            observation_context["previous_frame_ids"] = []
            observation_context["input_mode"] = "current_wrist_release_contact_only"
        verification = run_verification(
            backend,
            case["instruction"],
            case["success_criteria"],
            frame,
            case.get("failure_conditions", []),
            history=[] if is_lift else history,
            goal_reference=None if (
                is_lift or case["task"] in {
                    "turn_on_lightbulb", "turn_on_led", "turn_off_lightbulb",
                    "place_in_slider", "unstack_block"
                }
                or case["task"].startswith("rotate_")
            ) else goal_reference,
            goal_reference_wrist=goal_reference_wrist,
            initial_state=None if is_lift else initial_state,
            observation_context=observation_context,
            include_wrist=not (static_only or case["task"] == "move_slider_right"),
        )
        row = dict(case)
        row["verification"] = verification
        row["qwen_success"] = qwen_success(verification["result"])
        row["agreement"] = row["qwen_success"] == bool(row["oracle_success"])
        rows.append(row)
        output.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "manifest": str(manifest_path),
            "use_goal_reference": bool(use_goal_reference),
            "rows": rows,
        }
        output.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print("[{}/{}] {}".format(len(rows), len(manifest["cases"]), case["case_id"]), flush=True)
    report = {
        "manifest": str(manifest_path),
        "use_goal_reference": bool(use_goal_reference),
        "rows": rows,
    }
    report["summary"] = summarize(rows)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"cases": 0, "agreement": 0, "accuracy": None}
    agree = sum(bool(row["agreement"]) for row in rows)
    statuses: Dict[str, int] = {}
    for row in rows:
        status = row["verification"]["result"]["decision"]["status"]
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "cases": len(rows),
        "agreement": agree,
        "accuracy": agree / len(rows),
        "oracle_positive": sum(bool(row["oracle_success"]) for row in rows),
        "qwen_positive": sum(bool(row["qwen_success"]) for row in rows),
        "qwen_statuses": statuses,
        "mean_latency_seconds": sum(row["verification"]["total_latency_seconds"] for row in rows) / len(rows),
    }


def image_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return "data:image/png;base64," + encoded


def render_html(report: Dict[str, Any], manifest_path: Path, destination: Path) -> None:
    summary = report.get("summary") or summarize(report.get("rows", []))
    cards = []
    for row in report.get("rows", []):
        result = row["verification"]["result"]
        decision = result["decision"]
        observation = result["observation"]
        temporal = result["temporal_change"]
        reference_html = ""
        is_lift = row.get("task", "").startswith("lift_")
        if report.get("use_goal_reference") and row.get("goal_reference_static") and not is_lift:
            reference_html = (
                '<figure><img src="{}"><figcaption>Goal reference (static)</figcaption></figure>'.format(
                    image_uri(manifest_path.parent / row["goal_reference_static"])
                )
            )
        if (
            report.get("use_goal_reference")
            and row.get("goal_reference_wrist")
            and (is_lift or row.get("task") == "place_in_slider")
        ):
            reference_html += (
                '<figure><img src="{}"><figcaption>Goal reference (wrist)</figcaption></figure>'.format(
                    image_uri(manifest_path.parent / row["goal_reference_wrist"])
                )
            )
        current_static_path = row["static_image"]
        current_static_html = '<figure><img src="{}"><figcaption>Static</figcaption></figure>'.format(
            image_uri(manifest_path.parent / current_static_path)
        )
        initial_html = (
            '<figure><img src="{}"><figcaption>Initial static</figcaption></figure>'.format(
                image_uri(manifest_path.parent / row["initial_static_image"])
            )
        )
        history_html = "".join(
            '<figure><img src="{}"><figcaption>Recent static {}</figcaption></figure>'.format(
                image_uri(manifest_path.parent / item["static_image"]), index
            )
            for index, item in enumerate(row.get("history", []), 1)
        )
        if row.get("task") == "close_drawer":
            current_static_path = row["close_initial_current_static"]
            initial_html = ""
            history_html = ""
        elif row.get("task") in {
            "turn_on_lightbulb", "turn_off_lightbulb", "turn_on_led", "turn_off_led"
        }:
            initial_html = ""
            history_html = ""
            if row.get("task") in {"turn_on_lightbulb", "turn_on_led", "turn_off_lightbulb"}:
                reference_html = ""
        elif row.get("task") == "place_in_slider":
            initial_html = ""
            history_html = ""
            reference_html = (
                '<figure><img src="{}"><figcaption>Goal reference (wrist, released)</figcaption></figure>'.format(
                    image_uri(manifest_path.parent / row["goal_reference_wrist"])
                )
                if report.get("use_goal_reference") and row.get("goal_reference_wrist")
                else ""
            )
        elif row.get("task") == "move_slider_right":
            initial_html = ""
            history_html = ""
        elif row.get("task") == "unstack_block":
            initial_html = ""
            history_html = ""
            reference_html = ""
        elif row.get("task", "").startswith("rotate_"):
            current_static_html = ""
            initial_html = ""
            history_html = ""
            reference_html = ""
        wrist_html = ""
        if row.get("task") not in {
            "close_drawer", "turn_on_lightbulb", "turn_off_lightbulb",
            "turn_on_led", "turn_off_led", "move_slider_right"
        }:
            wrist_html = '<figure><img src="{}"><figcaption>Current wrist</figcaption></figure>'.format(
                image_uri(manifest_path.parent / row["wrist_image"])
            )
        criteria = "".join(
            "<li><b>{}</b> ({:.2f}, {}): {}</li>".format(
                html.escape(item["status"]), float(item["confidence"]),
                html.escape(item["observability"]), html.escape(item.get("evidence", ""))
            )
            for item in result["success_criteria"]
        )
        cards.append(
            """<article class="card {agreement}">
<h2>{case_id}</h2><p class="instruction">{instruction}</p>
<div class="images">{current_static}
{wrist}{initial}{history}{reference}</div>
<div class="verdict"><span>Oracle: <b>{oracle}</b></span><span>Qwen: <b>{qwen}</b></span>
<span>Confidence: {confidence:.2f}</span><span>Latency: {latency:.3f}s</span></div>
<p><b>Observation:</b> quality={view_quality}, occlusion={occlusion}, motion={motion}</p>
<p><b>Temporal:</b> progress={progress}, still_changing={still_changing} — {temporal_evidence}</p>
<p>{reason}</p><ol>{criteria}</ol></article>""".format(
                agreement="agree" if row["agreement"] else "disagree",
                case_id=html.escape(row["case_id"]), instruction=html.escape(row["instruction"]),
                current_static=current_static_html,
                wrist=wrist_html,
                reference=reference_html,
                initial=initial_html,
                history=history_html,
                oracle="success" if row["oracle_success"] else "not success",
                qwen=html.escape(decision["status"]), confidence=float(decision["confidence"]),
                latency=float(row["verification"]["total_latency_seconds"]),
                reason=html.escape(decision.get("reason", "")), criteria=criteria,
                view_quality=html.escape(str(observation.get("view_quality", ""))),
                occlusion=html.escape(str(observation.get("occlusion", ""))),
                motion=html.escape(str(observation.get("robot_motion_state", ""))),
                progress=html.escape(str(temporal.get("progress", ""))),
                still_changing=html.escape(str(temporal.get("still_changing", ""))),
                temporal_evidence=html.escape(str(temporal.get("evidence", ""))),
            )
        )
    accuracy = summary.get("accuracy")
    accuracy_text = "n/a" if accuracy is None else "{:.1%}".format(accuracy)
    document = """<!doctype html><html><head><meta charset="utf-8"><title>CALVIN-D Qwen Verifier</title>
<style>body{{font:15px system-ui;margin:24px;background:#f4f6f8;color:#18212b}}header{{position:sticky;top:0;background:#f4f6f8;padding:8px 0;z-index:2}}.summary{{display:flex;gap:24px;font-size:18px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:18px}}.card{{background:white;border:3px solid #d33;border-radius:12px;padding:16px}}.card.agree{{border-color:#2a6}}h2{{font-size:17px;margin:0}}.instruction{{color:#52606d}}.images{{display:grid;grid-template-columns:2fr 1fr;gap:10px}}figure{{margin:0}}img{{width:100%;height:230px;object-fit:contain;background:#111}}figcaption{{text-align:center}}.verdict{{display:flex;flex-wrap:wrap;gap:14px;margin-top:12px;padding:8px;background:#eef2f5}}li{{margin:6px 0}}</style></head><body>
<header><h1>CALVIN-D Qwen 子任务判定</h1><div class="summary"><span>Goal reference: {reference_mode}</span><span>Cases: {cases}</span><span>Agreement: {agree}</span><span>Accuracy: {accuracy}</span><span>Mean latency: {latency:.3f}s</span></div><p>绿色边框表示与 CALVIN Oracle 一致，红色表示不一致。图像已嵌入 HTML。</p></header>
<main class="grid">{cards}</main></body></html>""".format(
        cases=summary.get("cases", 0), agree=summary.get("agreement", 0), accuracy=accuracy_text,
        latency=float(summary.get("mean_latency_seconds") or 0),
        cards="".join(cards),
        reference_mode="enabled" if report.get("use_goal_reference") else "disabled",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("weights/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--use-goal-reference", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.output.read_text(encoding="utf-8")) if args.render_only else evaluate(
        args.manifest.resolve(), args.model_path, args.output.resolve(), args.max_new_tokens,
        args.use_goal_reference,
    )
    render_html(report, args.manifest.resolve(), args.html.resolve())
    print(json.dumps(report.get("summary") or summarize(report.get("rows", [])), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
