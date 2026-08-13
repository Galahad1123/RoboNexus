"""Small HTTP boundary exposing one live CALVIN Agent session."""

import json
import copy
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import RLock
from time import monotonic
from urllib.parse import urlparse


class CalvinToolSessionServer:
    def __init__(
        self,
        host,
        port,
        session,
        tools,
        artifacts,
        session_document,
        on_terminal=None,
        terminal_grace_seconds=30.0,
    ):
        self.session = session
        self.tools = {tool.name: tool for tool in tools}
        self.artifacts = artifacts
        self.session_document = dict(session_document)
        self.lock = RLock()
        self.on_terminal = on_terminal
        self.terminal_grace_seconds = float(terminal_grace_seconds)
        self._terminal_notified = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if urlparse(self.path).path != "/session":
                    return self._json(404, {"error": "not_found"})
                return self._json(200, owner.describe_session())

            def do_POST(self):
                path = urlparse(self.path).path
                if not path.startswith("/tools/"):
                    return self._json(404, {"error": "not_found"})
                tool_name = path[len("/tools/") :]
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    result = owner.call(tool_name, payload.get("params"))
                    return self._json(200, result)
                except (TypeError, ValueError, KeyError) as exc:
                    return self._json(
                        400,
                        {"error": type(exc).__name__, "message": str(exc)},
                    )
                except Exception as exc:
                    return self._json(
                        500,
                        {"error": type(exc).__name__, "message": str(exc)},
                    )

            def log_message(self, format_string, *args):
                del format_string, args

            def _json(self, status, payload):
                body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # CALVIN/PyBullet rendering is thread-affine: the environment and its
        # EGL context must be used on the thread that created them. A robot
        # Session is serialized already, so serving requests on this owner
        # thread is both correct and sufficient.
        self.httpd = HTTPServer((host, int(port)), Handler)

    def describe_session(self):
        document = dict(self.session_document)
        document.update(
            {
                "session_id": self.session.session_id,
                "session_status": self.session.status.value,
                "termination_reason": self.session.termination_reason,
                "latest_observation_id": self.session.latest_observation.observation_id,
                "budget": _budget_dict(self.session.budget_snapshot()),
                "tools": [
                    self._published_tool_spec(tool)
                    for tool in self.tools.values()
                ],
            }
        )
        return document

    def _published_tool_spec(self, tool):
        parameters = copy.deepcopy(tool.parameters)
        allowed = list(self.session_document.get("allowed_skill_ids", ()))
        vocabulary = list(self.session_document.get("semantic_vocabulary", ()))
        if tool.name in {"ground_targets", "execute_skill"} and allowed:
            parameters["properties"]["skill_id"]["enum"] = allowed
        if tool.name == "ground_targets":
            proposal = parameters["properties"]["proposals"]["items"]
            proposal["properties"] = {
                "role": proposal["properties"]["role"],
                "representation": {"type": "string", "enum": ["semantic"]},
                "semantic_id": {
                    "type": "string",
                    "enum": vocabulary,
                },
                "description": proposal["properties"]["description"],
            }
            proposal["required"] = [
                "role",
                "representation",
                "semantic_id",
                "description",
            ]
        description = tool.description
        if tool.name == "ground_targets":
            description += " This CALVIN session supports semantic proposals only."
        return {
            "name": tool.name,
            "description": description,
            "parameters": parameters,
        }

    def call(self, tool_name, params):
        try:
            tool = self.tools[tool_name]
        except KeyError:
            raise KeyError("unknown tool {!r}".format(tool_name))
        with self.lock:
            result = tool.call(params)
            payload = result.to_dict()
            payload["artifact_paths"] = [
                str(self.artifacts.resolve(item.uri))
                for item in result.artifacts
                if item.uri.startswith(("artifact://", "mask://"))
            ]
            payload["session_status"] = self.session.status.value
            self._notify_terminal()
            return payload

    def serve_forever(self):
        # Keep a short read-only window after terminal status so the Qwen
        # runner can fetch /session, then let the process enter its finally
        # block and release simulator resources automatically.
        self.httpd.timeout = 0.2
        terminal_since = None
        while True:
            self.httpd.handle_request()
            if self.session.status.terminal:
                self._notify_terminal()
                terminal_since = terminal_since or monotonic()
                if monotonic() - terminal_since >= self.terminal_grace_seconds:
                    return

    def _notify_terminal(self):
        if not self.session.status.terminal or self._terminal_notified:
            return
        self._terminal_notified = True
        if self.on_terminal is not None:
            self.on_terminal()

    def close(self):
        self.httpd.server_close()


def _budget_dict(budget):
    return {
        "tool_calls_used": budget.tool_calls_used,
        "tool_calls_remaining": budget.tool_calls_remaining,
        "environment_steps_used": budget.environment_steps_used,
        "environment_steps_remaining": budget.environment_steps_remaining,
        "replans_used": budget.replans_used,
        "replans_remaining": budget.replans_remaining,
    }
