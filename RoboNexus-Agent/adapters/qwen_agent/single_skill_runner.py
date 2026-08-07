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
verify_goal after every attempt. If verification fails, inspect again and choose
whether to reground, approach the source/target, or retry. For this first-version
single-skill test, after any failed execute_skill you MUST inspect_scene, ground
the required source/target on the fresh observation, call approach_target, then
inspect_scene and ground again before retrying execute_skill. Do not use
abort_goal with repeated_execution_failure until execute_skill reports attempt=3.
Never reuse a stale observation or grounding. A goal is complete only when verify_goal returns
data.status=success. If it cannot reasonably continue, call abort_goal. Keep
tool arguments concise and do not merely describe a tool call in prose."""


def run_single_skill_agent(
    runtime_url, model_path, seed=42, max_new_tokens=768, output=None
):
    session = fetch_session(runtime_url)
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
