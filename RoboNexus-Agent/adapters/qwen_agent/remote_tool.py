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


def build_remote_tools(runtime_url, session_document):
    return [
        RemoteRobotTool(runtime_url, spec)
        for spec in session_document["tools"]
    ]


def _qwen_compatible_schema(schema):
    """Adapt the strict protocol schema to qwen-agent 0.0.34's top-level subset."""
    return {
        "type": schema["type"],
        "properties": dict(schema.get("properties", {})),
        "required": list(schema.get("required", [])),
    }


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
