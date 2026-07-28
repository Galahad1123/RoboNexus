"""Small dependency-free JSON HTTP helpers for isolated Conda services."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


Route = Callable[[dict[str, Any]], dict[str, Any]]


def serve(host: str, port: int, routes: dict[str, Route]) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            route = routes.get(self.path)
            if route is None:
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                self._send(200, route(payload))
            except (ValueError, KeyError, TypeError) as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:  # service boundary: return an actionable error
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    ThreadingHTTPServer((host, port), Handler).serve_forever()


def post_json(url: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        # Model services are normally localhost/LAN endpoints. Ignore inherited
        # HTTP_PROXY settings, which otherwise route localhost through a proxy.
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {detail}") from exc
    if "error" in result:
        raise RuntimeError(f"{url}: {result['error']}")
    return result
