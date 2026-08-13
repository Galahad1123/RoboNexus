"""Qwen-Agent/Qwen3-VL driver for one remote robot goal session."""

import json
from pathlib import Path

from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import ContentItem, Message

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
):
    session = fetch_session(runtime_url)
    if expected_skill_ids is not None and list(expected_skill_ids) != session["allowed_skill_ids"]:
        raise RuntimeError(
            "connected Runtime exposes skills {}, expected {}".format(
                session["allowed_skill_ids"], list(expected_skill_ids)
            )
        )
    assistant = Assistant(
        function_list=build_remote_tools(runtime_url, session),
        llm={
            "model": str(model_path),
            "model_type": "transformers",
            "device": "cuda",
            "generate_cfg": {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": False,
                "fncall_prompt_type": "nous",
            },
        },
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
    snapshots = 0
    for response in assistant.run(messages=messages, lang="en", seed=int(seed)):
        final_messages = response
        snapshots += 1
    final_session = fetch_session(runtime_url)
    report = {
        "session_before": session,
        "session_after": final_session,
        "final_messages": [_message_dict(item) for item in final_messages],
        "num_agent_updates": snapshots,
    }
    destination = Path(output) if output else Path(session["output_dir"]) / "qwen_agent_result.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    return report


def _message_dict(message):
    return message.model_dump() if hasattr(message, "model_dump") else dict(message)
