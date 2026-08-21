#!/usr/bin/env python3
"""Isolated Qwen3-VL subtask verifier experiment for CALVIN.

This module deliberately has no dependency on RoboNexus Runtime. It evaluates
one static-camera/wrist-camera pair against the task initial state first and,
whenever recent frames are available, always performs sparse temporal verification.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image


DECISION_STATUSES = {"success", "in_progress", "failure", "need_more_evidence"}
CRITERION_STATES = {"pass", "fail", "unknown"}
OBSERVABILITY = {"observable", "partially_observable", "occluded", "insufficient_history"}
CAMERAS = {"static", "wrist", "both", "temporal", "none"}
PROGRESS_STATES = {"positive", "negative", "none", "uncertain"}

OBSERVABILITY_ALIASES = {
    "visible": "observable",
    "clear": "observable",
    "sufficient": "observable",
    "fully_observable": "observable",
    "partial": "partially_observable",
    "partially_visible": "partially_observable",
    "partially_observed": "partially_observable",
    "blocked": "occluded",
    "severely_occluded": "occluded",
    "insufficient": "insufficient_history",
    "not_observable": "insufficient_history",
    "unobservable": "insufficient_history",
}

CRITERION_STATE_ALIASES = {
    "passed": "pass",
    "satisfied": "pass",
    "true": "pass",
    "failed": "fail",
    "unsatisfied": "fail",
    "false": "fail",
    "uncertain": "unknown",
    "not_sure": "unknown",
    "insufficient_evidence": "unknown",
}

EVIDENCE_SOURCE_ALIASES = {
    "current_static": "static",
    "static_view": "static",
    "current_wrist": "wrist",
    "wrist_view": "wrist",
    "static_and_wrist": "both",
    "recent_and_current": "temporal",
    "history": "temporal",
}

SYSTEM_PROMPT = """You are a conservative verifier for a robot manipulation subtask.
Judge only facts that are observable in the supplied STATIC and WRIST camera images.
The static camera is usually best for global spatial relations. The wrist camera is
usually best for local gripper/object relations. Images with the same time label are
synchronized. Older images are sparse observations for resolving ambiguity, not a
successful reference trajectory.
When INITIAL_STATIC_VIEW and INITIAL_WRIST_VIEW are supplied, they show the state at
the beginning of this task. Compare them with CURRENT views to identify which object
and state changed. The initial state is context, never evidence that the goal is done.
When a GOAL_REFERENCE_STATIC_VIEW is supplied, it is a successful example from a
different demonstration. Use it to identify task-relevant objects and understand the
semantic goal state. Do not require pixel-perfect matching, identical robot pose, or
identical positions of unrelated objects.
For close_drawer, exactly two STATIC comparison images are supplied and each contains
only the lower half of its original frames. INITIAL_TOP_CURRENT_BOTTOM has the task's
initial state on TOP and current state on BOTTOM. GOAL_REFERENCE_STATIC_VIEW has a
successfully CLOSED empty drawer on TOP and an OPEN empty drawer on BOTTOM. Do not compare
the halves as simultaneous scene regions; they are different states.
For turn_on_lightbulb, exactly one enlarged STATIC crop is supplied: CURRENT, taken from
the original upper-right quarter. Judge the task only by whether a small yellow sphere is
visible near the top of this current image. No goal, initial, recent, wrist, switch state,
or switch-related reasoning is used for this task.
For turn_on_led, exactly one enlarged STATIC crop is supplied: CURRENT, taken from the
original upper-left quarter. Pass exactly when a green rectangle is visible in its upper
part; otherwise fail/in_progress. The turn_off_led task may also include a GOAL crop.
For turn_off_lightbulb, exactly one enlarged STATIC crop is supplied: CURRENT, taken from
the original upper-right quarter. Pass exactly when no yellow circular object is visible
in its upper part; a yellow circle means fail/in_progress. Do not reason about a switch.
Use this fixed landmark guide for CALVIN STATIC views when locating small or ambiguous
facilities:
- drawer: in the lower-right area of the image; its front has a gray U-shaped handle.
  A closed drawer front is flush with the cabinet and has no visible interior gap. Do not
  mistake the tabletop, cabinet panels, or space around the handle for drawer interior.
- slider: its sliding door has a vertical gray handle. When the door panel is around the
  middle of the table/cabinet, the slider is in the RIGHT state. It is in the LEFT state
  only when the door panel itself has moved to the left side of its travel range. The
  recessed compartment revealed behind this door is the slider placement surface; it is
  distinct from the main tabletop.
- LED: a flat rectangular object in the upper-left of the image, above the slider. It
  appears as a plain white rectangle when off and a green rectangle when on. For the LED
  tasks, visible green at that landmark means ON; absence of green there means OFF.
- lightbulb: in a lightbulb crop, a small sphere near the top. It is yellow when on and
  non-yellow/white when off. Judge that sphere directly; do not use or discuss any switch.
For place_in_slider in this experiment, CURRENT STATIC/WRIST plus a successfully released
GOAL_REFERENCE_WRIST_VIEW are supplied. Treat the manipulated item as a rectangular
object. Separately judge that this rectangular object is visibly placed on a table surface
in CURRENT_STATIC_VIEW and that CURRENT_WRIST_VIEW shows no grasped rectangular object
when compared with the empty/released goal wrist.
For push_into_drawer, track the block from the initial view to the open lower-right drawer.
Do not count a block merely near the drawer rim or still on the main tabletop as inside.
For unstack_block, only CURRENT_STATIC_VIEW and CURRENT_WRIST_VIEW are supplied. Judge
the current spatial relation independently in each camera. Report separate criterion
results for STATIC and WRIST: each must show no block/cube overlapping and resting on
another block/cube. Do not infer an earlier stack or use arm pose alone.
For every rotate task, only CURRENT_WRIST_VIEW is supplied. The two thin, pointed objects
at the top of the wrist image are the two gripper fingers. If either finger visibly contacts
a colored block/cube/rectangle, the object is not released and the criterion must fail or
remain in_progress. If neither finger contacts such an object, pass. Ignore rotation.
This guide identifies object locations only. Always use the supplied images to judge the
current state.
For lift tasks, first inspect CURRENT_WRIST_VIEW itself for the requested colored cube,
then use GOAL_REFERENCE_WRIST_VIEW only to compare the holding relation. Pass only when
the colored cube is visibly between the gripper fingers in CURRENT_WRIST_VIEW and matches
the successful reference relation.
Gripper proximity, fingers surrounding empty space, or a cube visible elsewhere is not a
successful grasp.

Evaluate every success criterion separately. A criterion marked temporal_required must
use the ordered RECENT and CURRENT views. Stable means the relevant goal state changes
negligibly across those views; it does not require a long pause or a particular robot
pose. If a critical fact is occluded, ambiguous, or contradicted across cameras, mark it
unknown. Do not infer success from the instruction or from an action that appears likely.
The criterion status must agree with its own evidence text. For example, evidence saying
"drawer is open, not closed" must never receive pass for a close-drawer criterion.
Return exactly one JSON object:
{
  "subtask_id": "the supplied subtask id",
  "observation": {
    "current_frame_id": 0,
    "previous_frame_ids": [],
    "robot_motion_state": "stopped|moving|uncertain",
    "view_quality": "sufficient|partial|insufficient",
    "occlusion": "none|partial|severe",
    "confidence": 0.0
  },
  "success_criteria": [
    {"criterion_id": "criterion id",
     "observability": "observable|partially_observable|occluded|insufficient_history",
     "status": "pass|fail|unknown", "confidence": 0.0,
     "evidence_source": "static|wrist|both|temporal|none",
     "evidence": "short visible evidence"}
  ],
  "temporal_change": {
    "progress": "positive|negative|none|uncertain",
    "still_changing": false, "unexpected_change": false,
    "confidence": 0.0, "evidence": "short temporal evidence"
  },
  "decision": {
    "status": "success|in_progress|failure|need_more_evidence",
    "confidence": 0.0,
    "recommended_action": "terminate|continue|retry|observe_again",
    "reason": "short overall reason"
  }
}

Use success only when every required criterion passes. Use in_progress when a required
criterion visibly fails but normal continuation can resolve it. Use failure only for an
explicit failure condition or visibly unrecoverable outcome. Use need_more_evidence when
a required criterion is unknown. Confidence values must be in [0,1].
"""


@dataclass(frozen=True)
class FramePair:
    static: Optional[Image.Image]
    wrist: Optional[Image.Image]
    label: str = "current"


class QwenVerifierBackend:
    """Small direct Transformers backend, kept local to this experiment."""

    def __init__(self, model_path: Path, max_new_tokens: int = 1024) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_path), dtype=torch.bfloat16, device_map="auto"
        ).eval()
        self.max_new_tokens = int(max_new_tokens)

    def verify(
        self,
        subtask: str,
        success_criteria: Sequence[Dict[str, Any]],
        failure_conditions: Sequence[str],
        frames: Sequence[FramePair],
        goal_reference: Optional[Image.Image] = None,
        goal_reference_wrist: Optional[Image.Image] = None,
        initial_state: Optional[FramePair] = None,
        observation_context: Optional[Dict[str, Any]] = None,
        include_wrist: bool = True,
    ) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        if initial_state is not None:
            content.extend(
                [
                    {"type": "text", "text": "INITIAL_STATIC_VIEW"},
                    {"type": "image", "image": initial_state.static},
                ]
            )
            if include_wrist:
                if initial_state.wrist is None:
                    raise ValueError("initial wrist image is required when include_wrist=True")
                content.extend(
                    [
                        {"type": "text", "text": "INITIAL_WRIST_VIEW"},
                        {"type": "image", "image": initial_state.wrist},
                    ]
                )
        for frame in frames:
            if frame.static is not None:
                content.extend(
                    [
                        {"type": "text", "text": "{}_STATIC_VIEW".format(frame.label.upper())},
                        {"type": "image", "image": frame.static},
                    ]
                )
            if include_wrist:
                if frame.wrist is None:
                    raise ValueError("wrist image is required when include_wrist=True")
                content.extend(
                    [
                        {"type": "text", "text": "{}_WRIST_VIEW".format(frame.label.upper())},
                        {"type": "image", "image": frame.wrist},
                    ]
                )
        if goal_reference is not None:
            content.extend(
                [
                    {"type": "text", "text": "GOAL_REFERENCE_STATIC_VIEW"},
                    {"type": "image", "image": goal_reference},
                ]
            )
        if goal_reference_wrist is not None:
            content.extend(
                [
                    {"type": "text", "text": "GOAL_REFERENCE_WRIST_VIEW"},
                    {"type": "image", "image": goal_reference_wrist},
                    {
                        "type": "text",
                        "text": (
                            "Use this successful wrist-view example to identify the expected "
                            "gripper-object relation (held or released according to the task); "
                            "do not require an identical arm pose."
                        ),
                    },
                ]
            )
        payload = {
            "subtask_id": subtask,
            "subtask": subtask,
            "observation_context": observation_context or {},
            "required_success_criteria": list(success_criteria),
            "explicit_failure_conditions": list(failure_conditions),
        }
        content.append(
            {"type": "text", "text": "VERIFICATION_REQUEST:\n" + json.dumps(payload, ensure_ascii=False)}
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        attempts = [self.max_new_tokens, max(1536, self.max_new_tokens * 2)]
        errors = []
        for attempt, token_limit in enumerate(attempts, 1):
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs, max_new_tokens=token_limit, do_sample=False
                )
            trimmed = generated[:, inputs["input_ids"].shape[1] :]
            text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            try:
                return parse_json_object(text)
            except (ValueError, json.JSONDecodeError) as error:
                errors.append(
                    "attempt {} (max_new_tokens={}): {}; raw={}".format(
                        attempt, token_limit, error, repr(text[:4000])
                    )
                )
        raise ValueError("Qwen returned invalid JSON after automatic retry:\n" + "\n".join(errors))


def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse the first complete JSON object, tolerating fences or trailing prose."""

    start = text.find("{")
    if start < 0:
        raise ValueError("verifier returned no JSON object: {}".format(text[:300]))
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("verifier response must be a JSON object")
    return value


def _confidence(value: Any, field: str) -> float:
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("{} must be in [0, 1]".format(field))
    return confidence


def _canonical(value: Any, allowed: set, aliases: Dict[str, str]) -> Optional[str]:
    normalized = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in allowed else None


def validate_result(raw: Dict[str, Any], expected_criteria: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate and normalize the model contract; reject internally invalid success."""

    result = dict(raw)
    observation = result.get("observation")
    temporal = result.get("temporal_change")
    decision = result.get("decision")
    if not all(isinstance(item, dict) for item in (observation, temporal, decision)):
        raise ValueError("result requires observation, temporal_change, and decision objects")
    if decision.get("status") not in DECISION_STATUSES:
        raise ValueError("invalid decision status: {!r}".format(decision.get("status")))
    decision["confidence"] = _confidence(decision.get("confidence"), "decision confidence")
    observation["confidence"] = _confidence(
        observation.get("confidence"), "observation confidence"
    )
    if temporal.get("progress") not in PROGRESS_STATES:
        raise ValueError("invalid temporal progress: {!r}".format(temporal.get("progress")))
    temporal["confidence"] = _confidence(temporal.get("confidence"), "temporal confidence")

    expected_ids = [str(item["criterion_id"]) for item in expected_criteria]
    expected_by_id = {str(item["criterion_id"]): item for item in expected_criteria}
    criteria = result.get("success_criteria")
    if not isinstance(criteria, list):
        raise ValueError("success_criteria must be a list")
    by_id = {
        str(item.get("criterion_id")): dict(item) for item in criteria if isinstance(item, dict)
    }
    if set(by_id) != set(expected_ids):
        raise ValueError("criterion ids {} do not match expected {}".format(sorted(by_id), sorted(expected_ids)))

    normalized = []
    for criterion_id in expected_ids:
        item = by_id[criterion_id]
        raw_status = item.get("status")
        status = _canonical(raw_status, CRITERION_STATES, CRITERION_STATE_ALIASES)
        raw_observability = item.get("observability")
        observability = _canonical(
            raw_observability, OBSERVABILITY, OBSERVABILITY_ALIASES
        )
        raw_source = item.get("evidence_source")
        evidence_source = _canonical(raw_source, CAMERAS, EVIDENCE_SOURCE_ALIASES)
        corrections = []
        if status is None:
            status = "unknown"
            corrections.append("unrecognized status {!r} downgraded to unknown".format(raw_status))
        if observability is None:
            observability = (
                "insufficient_history"
                if expected_by_id[criterion_id].get("temporal_required", False)
                else "partially_observable"
            )
            status = "unknown"
            corrections.append(
                "unrecognized observability {!r} downgraded to {}".format(
                    raw_observability, observability
                )
            )
        if evidence_source is None:
            evidence_source = "none"
            corrections.append("unrecognized evidence_source {!r} mapped to none".format(raw_source))
        item["status"] = status
        item["observability"] = observability
        item["evidence_source"] = evidence_source
        if corrections:
            item["schema_corrections"] = corrections
        item["criterion_id"] = criterion_id
        item["confidence"] = _confidence(
            item.get("confidence"), "criterion {} confidence".format(criterion_id)
        )
        evidence = str(item.get("evidence", "")).lower()
        contradictions = expected_by_id[criterion_id].get("fail_if_evidence_contains", [])
        matched = [phrase for phrase in contradictions if phrase.lower() in evidence]
        if item["status"] == "pass" and matched:
            item["status"] = "fail"
            item["contract_correction"] = (
                "pass contradicted by evidence phrase(s): " + ", ".join(matched)
            )
        normalized.append(item)

    required_ids = {
        str(item["criterion_id"]) for item in expected_criteria if item.get("required", True)
    }
    required_results = [item for item in normalized if item["criterion_id"] in required_ids]
    if decision["status"] == "success" and any(item["status"] == "fail" for item in required_results):
        decision["status"] = "in_progress"
        decision["reason"] = "A required criterion failed after evidence consistency checks."
        decision["recommended_action"] = "continue"
    elif decision["status"] == "success" and any(
        item["status"] == "unknown" for item in required_results
    ):
        # A false-success caused by inconsistent model JSON must never cross the API boundary.
        decision["status"] = "need_more_evidence"
        decision["reason"] = "Model claimed success without passing every required criterion."
        decision["recommended_action"] = "observe_again"
    result["observation"] = observation
    result["temporal_change"] = temporal
    result["decision"] = decision
    result["success_criteria"] = normalized
    return result


def run_verification(
    backend: Any,
    subtask: str,
    success_criteria: Sequence[Dict[str, Any]],
    current: FramePair,
    failure_conditions: Sequence[str] = (),
    history: Sequence[FramePair] = (),
    max_history_pairs: int = 2,
    goal_reference: Optional[Image.Image] = None,
    goal_reference_wrist: Optional[Image.Image] = None,
    initial_state: Optional[FramePair] = None,
    observation_context: Optional[Dict[str, Any]] = None,
    include_wrist: bool = True,
) -> Dict[str, Any]:
    """Use one temporal call by default, falling back to current-only without history."""

    passes = []
    started = time.perf_counter()
    selected_history = list(history)[-max(0, int(max_history_pairs)) :]
    frames = selected_history + [current]
    inference_started = time.perf_counter()
    final = validate_result(
        backend.verify(
            subtask,
            success_criteria,
            failure_conditions,
            frames,
            goal_reference=goal_reference,
            goal_reference_wrist=goal_reference_wrist,
            initial_state=initial_state,
            observation_context=observation_context,
            include_wrist=include_wrist,
        ),
        success_criteria,
    )
    passes.append(
        {
            "mode": "sparse_temporal" if selected_history else "current_pair",
            "latency_seconds": time.perf_counter() - inference_started,
            "frame_pairs": len(frames),
            "result": final,
        }
    )

    return {
        "result": final,
        "used_temporal_recheck": bool(selected_history),
        "total_latency_seconds": time.perf_counter() - started,
        "passes": passes,
    }


def load_spec(path: Path) -> Dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec.get("subtask"), str) or not spec["subtask"].strip():
        raise ValueError("spec requires a non-empty subtask")
    criteria = spec.get("success_criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("spec requires non-empty success_criteria")
    for position, item in enumerate(criteria, 1):
        if not isinstance(item, dict) or not item.get("description"):
            raise ValueError("each success criterion requires a description")
        item.setdefault("criterion_id", item.pop("id", str(position)))
        item["criterion_id"] = str(item["criterion_id"])
        item.setdefault("required", True)
        item.setdefault("evidence_source", "both")
        item.setdefault("temporal_required", False)
    spec.setdefault("failure_conditions", [])
    return spec


def load_pair(static_path: Path, wrist_path: Path, label: str) -> FramePair:
    return FramePair(Image.open(static_path).convert("RGB"), Image.open(wrist_path).convert("RGB"), label)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("weights/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--wrist", type=Path, required=True)
    parser.add_argument("--history-static", type=Path, action="append", default=[])
    parser.add_argument("--history-wrist", type=Path, action="append", default=[])
    parser.add_argument("--max-history-pairs", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--goal-reference-static", type=Path)
    parser.add_argument("--goal-reference-wrist", type=Path)
    parser.add_argument("--initial-static", type=Path)
    parser.add_argument("--initial-wrist", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.history_static) != len(args.history_wrist):
        parser.error("--history-static and --history-wrist counts must match")
    if bool(args.initial_static) != bool(args.initial_wrist):
        parser.error("--initial-static and --initial-wrist must be supplied together")

    spec = load_spec(args.spec)
    current = load_pair(args.static, args.wrist, "current")
    history = [
        load_pair(static, wrist, "history_{}".format(index))
        for index, (static, wrist) in enumerate(zip(args.history_static, args.history_wrist), 1)
    ]
    backend = QwenVerifierBackend(args.model_path, args.max_new_tokens)
    goal_reference = (
        Image.open(args.goal_reference_static).convert("RGB") if args.goal_reference_static else None
    )
    goal_reference_wrist = (
        Image.open(args.goal_reference_wrist).convert("RGB") if args.goal_reference_wrist else None
    )
    initial_state = (
        load_pair(args.initial_static, args.initial_wrist, "initial") if args.initial_static else None
    )
    report = run_verification(
        backend,
        spec["subtask"],
        spec["success_criteria"],
        current,
        spec["failure_conditions"],
        history,
        args.max_history_pairs,
        goal_reference,
        goal_reference_wrist,
        initial_state,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
