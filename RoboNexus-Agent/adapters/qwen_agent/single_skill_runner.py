"""Qwen-Agent/Qwen3-VL driver for one remote robot goal session."""

import json
from pathlib import Path

from qwen_agent.agents import Assistant
from qwen_agent.llm import get_chat_model
from qwen_agent.llm.schema import ContentItem, Message

from core.logging import atomic_write_json, read_json

from .remote_tool import build_remote_tools, fetch_session


SYSTEM_PROMPT = """You are the high-level planner for one robot goal.
Use the provided tools; never emit low-level robot actions. Only choose a skill
from allowed_skill_ids. Inspect the current scene, ground every role required by
the selected skill. This CALVIN session supports semantic grounding only: use
representation="semantic", an exact semantic_id from semantic_vocabulary, and
the current observation_id; never send bbox, point, or duplicate roles. Always
include skill_id in ground_targets. Execute exactly one atomic skill attempt, and call
verify_goal whenever execute_skill reports atomic success; goal verification can
still fail when another subtask remains, in which case call inspect_scene before
planning and grounding the next allowed skill. After a failed execute_skill, always inspect and semantically reground
before retrying. Follow ROBOT_GOAL_CONTEXT.recovery_policy exactly. Do not use
abort_goal with repeated_execution_failure before the failed skill reports attempt=3.
Never reuse a stale observation or grounding. A goal is complete only when verify_goal returns
data.status=success. If it cannot reasonably continue, call abort_goal. Keep
tool arguments concise and do not merely describe a tool call in prose."""


def run_single_skill_agent(
    runtime_url,
    model_path,
    seed=42,
    max_new_tokens=768,
    output=None,
    expected_skill_ids=None,
    llm=None,
    max_agent_rounds=4,
):
    session = fetch_session(runtime_url)
    destination = Path(output) if output else Path(session["output_dir"]) / "qwen_agent_result.json"
    checkpoint = destination.with_name("qwen_agent_checkpoint.json")
    if expected_skill_ids is not None and list(expected_skill_ids) != session["allowed_skill_ids"]:
        raise RuntimeError(
            "connected Runtime exposes skills {}, expected {}".format(
                session["allowed_skill_ids"], list(expected_skill_ids)
            )
        )
    assistant = Assistant(
        function_list=build_remote_tools(runtime_url, session),
        llm=llm or build_qwen_llm(model_path, max_new_tokens),
        system_message=SYSTEM_PROMPT,
        name="RoboNexusAgent",
        description="Plans one robot skill at a time through bounded tools.",
    )
    initial = session["initial_observation"]
    context = {
        "goal_id": session["goal_id"],
        "goal_instruction": session["goal_instruction"],
        "success_spec": session["success_spec"],
        "allowed_skill_ids": session["allowed_skill_ids"],
        "semantic_vocabulary": session["semantic_vocabulary"],
        "required_grounding_roles": session["required_grounding_roles"],
        "approach_enabled": session["approach_enabled"],
        "recovery_policy": (
            "inspect, ground source/target, approach_target, inspect, reground, retry"
            if session["approach_enabled"]
            else "approach_target is deactivated; inspect, semantic reground, retry"
        ),
    }
    messages = [
        Message(
            role="user",
            content=[
                ContentItem(text="ROBOT_GOAL_CONTEXT:\n" + json.dumps(context, ensure_ascii=False)),
                ContentItem(text="CURRENT_STATIC_VIEW"),
                ContentItem(image=initial["static_image"]),
                ContentItem(text="CURRENT_WRIST_VIEW"),
                ContentItem(image=initial["wrist_image"]),
                ContentItem(text="GOAL_REFERENCE_VIEW: not provided"),
            ],
        )
    ]
    final_messages = []
    transcript = []
    snapshots = 0
    caught = None
    try:
        for agent_round in range(1, int(max_agent_rounds) + 1):
            final_messages = []
            for response in assistant.run(messages=messages, lang="en", seed=int(seed)):
                final_messages = response
                snapshots += 1
                atomic_write_json(
                    checkpoint,
                    _report(
                        session,
                        None,
                        transcript + list(final_messages),
                        snapshots,
                        "running",
                        agent_round,
                    ),
                )
            transcript.extend(final_messages)
            current_session = fetch_session(runtime_url)
            if _terminal(current_session.get("session_status")):
                break
            if agent_round >= int(max_agent_rounds):
                raise RuntimeError(
                    "Qwen stopped while robot Session remained {} after {} Agent rounds".format(
                        current_session.get("session_status"), agent_round
                    )
                )
            messages = [
                Message(
                    role="user",
                    content=_continuation_content(context, current_session, transcript),
                )
            ]
    except BaseException as error:
        caught = error
    try:
        final_session = fetch_session(runtime_url)
    except Exception as error:
        final_session = None
        if caught is None:
            caught = error
    runtime_result = read_json(destination.parent / "runtime_result.json")
    report = _report(
        session,
        final_session,
        transcript + ([] if transcript[-len(final_messages):] == final_messages else list(final_messages)) if final_messages else transcript,
        snapshots,
        "completed" if caught is None else "interrupted",
        agent_round if "agent_round" in locals() else 0,
    )
    report["runtime_result"] = runtime_result
    if caught is not None:
        report["error"] = {"type": type(caught).__name__, "message": str(caught)}
    effective_status = (runtime_result or {}).get("session_status") or (
        (final_session or {}).get("session_status")
    )
    report["effective_status"] = effective_status or "unknown"
    if _terminal(effective_status):
        report["run_state"] = "completed"
    atomic_write_json(destination, report)
    if caught is not None and not _terminal(effective_status):
        raise caught
    return report


def build_qwen_llm(model_path, max_new_tokens=768):
    """Load one reusable Qwen model for single or batched robot sessions."""

    return get_chat_model(
        {
            "model": str(model_path),
            "model_type": "transformers",
            "device": "cuda",
            "generate_cfg": {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": False,
                "fncall_prompt_type": "nous",
            },
        }
    )


def _terminal(status):
    return status == "succeeded" or str(status).startswith("aborted")


def _continuation_content(context, current_session, transcript):
    """Compact a completed Agent round while retaining the newest visual state."""

    summary = []
    latest_visual = []
    for message in transcript:
        if message.get("role") != "function":
            continue
        content = message.get("content")
        items = content if isinstance(content, list) else []
        serialized = content
        if items:
            serialized = next(
                (item.get("text") for item in items if isinstance(item, dict) and item.get("text", "").startswith("{")),
                None,
            )
        try:
            result = json.loads(serialized) if isinstance(serialized, str) else serialized
        except json.JSONDecodeError:
            continue
        if not isinstance(result, dict):
            continue
        data = result.get("data") or {}
        summary.append(
            {
                "tool": message.get("name"),
                "status": result.get("status"),
                "skill_id": data.get("skill_id"),
                "logical_subtask_id": data.get("logical_subtask_id"),
                "attempt": data.get("attempt"),
                "success": data.get("success"),
                "termination_reason": data.get("termination_reason") or data.get("reason"),
                "observation_id": data.get("after_observation_id") or (data.get("observation") or {}).get("observation_id"),
            }
        )
        if message.get("name") == "inspect_scene" and items:
            latest_visual = [ContentItem(**item) for item in items[1:] if isinstance(item, dict)]
    payload = {
        "goal_context": context,
        "current_session": {
            "session_status": current_session.get("session_status"),
            "latest_observation_id": current_session.get("latest_observation_id"),
            "budget": current_session.get("budget"),
        },
        "tool_summary": summary[-16:],
    }
    content = [
        ContentItem(text="CONTINUE_ROBOT_GOAL_COMPACTED:\n" + json.dumps(payload, ensure_ascii=False)),
        ContentItem(
            text=(
                "Continue from the latest state. Do not repeat an exhausted attempt; verify, choose "
                "the next atomic skill, or abort according to the recovery policy."
            )
        ),
    ]
    content.extend(latest_visual)
    return content


def _report(session, final_session, messages, snapshots, run_state, agent_round=1):
    return {
        "run_state": run_state,
        "session_before": session,
        "session_after": final_session,
        "final_messages": [_message_dict(item) for item in messages],
        "num_agent_updates": snapshots,
        "num_agent_rounds": agent_round,
    }


def _message_dict(message):
    return message.model_dump() if hasattr(message, "model_dump") else dict(message)
