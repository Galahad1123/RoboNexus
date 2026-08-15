"""Qwen-Agent BaseTool adapters for a remote robot Session Runtime."""

import json
from urllib.request import ProxyHandler, Request, build_opener

from qwen_agent.llm.schema import ContentItem
from qwen_agent.tools.base import BaseTool


_DIRECT_OPENER = build_opener(ProxyHandler({}))


class RemoteRobotTool(BaseTool):
    def __init__(self, runtime_url, tool_spec, timeout=600.0):
        self.name = tool_spec["name"]
        self.description = tool_spec["description"]
        self.parameters = _qwen_compatible_schema(tool_spec["parameters"])
        self.runtime_url = runtime_url.rstrip("/")
        self.timeout = float(timeout)
        super().__init__()

    def call(self, params, **kwargs):
        del kwargs
        parsed = self._verify_json_format_args(params)
        parsed = _normalize_tool_params(self.name, parsed)
        payload = json.dumps({"params": parsed}).encode("utf-8")
        request = Request(
            "{}/tools/{}".format(self.runtime_url, self.name),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _DIRECT_OPENER.open(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        return _result_content(self.name, result)


def fetch_session(runtime_url, timeout=30.0):
    with _DIRECT_OPENER.open(
        runtime_url.rstrip("/") + "/session", timeout=timeout
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def call_remote_tool(runtime_url, tool_name, params, timeout=600.0):
    """Call a Runtime tool directly without consuming a Qwen-Agent round."""

    payload = json.dumps({"params": params}).encode("utf-8")
    request = Request(
        "{}/tools/{}".format(runtime_url.rstrip("/"), tool_name),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _DIRECT_OPENER.open(request, timeout=float(timeout)) as response:
        return json.loads(response.read().decode("utf-8"))


def build_remote_tools(runtime_url, session_document):
    return [
        RemoteRobotTool(runtime_url, spec)
        for spec in session_document["tools"]
    ]


def _qwen_compatible_schema(schema):
    """Adapt the strict protocol schema to qwen-agent 0.0.34's top-level subset."""
    result = {
        "type": schema["type"],
        "properties": dict(schema.get("properties", {})),
        "required": list(schema.get("required", [])),
    }
    # Qwen3-VL reliably identifies role/representation/semantic_id but can
    # omit this purely explanatory field. Keep the backend protocol strict and
    # synthesize the description at the adapter boundary instead of spending
    # an Agent turn repairing semantically complete grounding arguments.
    proposals = result["properties"].get("proposals")
    if isinstance(proposals, dict):
        item = proposals.get("items")
        if isinstance(item, dict) and "required" in item:
            item = dict(item)
            item["required"] = [
                field for field in item["required"] if field != "description"
            ]
            proposals = dict(proposals)
            proposals["items"] = item
            result["properties"]["proposals"] = proposals
    return result


def _normalize_tool_params(tool_name, params):
    if tool_name != "ground_targets" or not isinstance(params, dict):
        return params
    normalized = dict(params)
    proposals = []
    for proposal in normalized.get("proposals", ()):
        proposal = dict(proposal)
        proposal.setdefault(
            "description",
            "Agent-selected {} {}".format(
                proposal.get("role", "object"),
                proposal.get("semantic_id", "target"),
            ),
        )
        proposals.append(proposal)
    normalized["proposals"] = proposals
    return normalized


def _result_content(tool_name, result):
    """Return JSON plus fresh visual observations for multimodal tool turns."""
    serialized = json.dumps(result, ensure_ascii=False, sort_keys=True)
    if tool_name != "inspect_scene" or result.get("status") != "succeeded":
        return serialized

    paths_by_kind = {
        artifact.get("kind"): path
        for artifact, path in zip(
            result.get("artifacts", ()), result.get("artifact_paths", ())
        )
    }
    static_path = paths_by_kind.get("current_static")
    wrist_path = paths_by_kind.get("current_wrist")
    if not static_path or not wrist_path:
        return serialized
    return [
        ContentItem(text=serialized),
        ContentItem(text="CURRENT_STATIC_VIEW (fresh inspect_scene observation)"),
        ContentItem(image=static_path),
        ContentItem(text="CURRENT_WRIST_VIEW (fresh inspect_scene observation)"),
        ContentItem(image=wrist_path),
    ]
