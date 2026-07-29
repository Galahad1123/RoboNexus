"""Python 3.8-compatible client for the generic MultiTask DiT server."""

import json
import logging
import os
import time
from typing import Any, Dict, Optional

import numpy as np


class ModelClient:
    def __init__(
        self,
        host="127.0.0.1",
        port=10093,
        connect_timeout=300.0,
    ):
        self.uri = "ws://{}:{}".format(host, port)
        self.request_index = 0
        self.websocket, self.metadata = self._wait_for_server(connect_timeout)

    def reset(self, episode_id, task, replan_interval):
        response = self._request(
            {
                "type": "reset",
                "episode_id": str(episode_id),
                "task": str(task),
                "replan_interval": int(replan_interval),
            }
        )
        _raise_for_error(response)
        return response

    def step(self, episode_id, observation, step):
        response = self._request(
            {
                "type": "step",
                "episode_id": str(episode_id),
                "step": int(step),
                "observation": observation,
            }
        )
        _raise_for_error(response)
        # msgpack-numpy may decode ndarray buffers as read-only views. CALVIN's
        # relative-action conversion scales slices in place, so ownership and
        # writability are part of the client boundary contract.
        action = np.array(response["action"], dtype=np.float32, copy=True)
        return action, response.get("metadata", {})

    def health(self):
        response = self._request({"type": "health"})
        _raise_for_error(response)
        return response

    def close(self):
        try:
            self.websocket.close()
        except Exception:
            pass

    def _request(self, payload):
        self.request_index += 1
        payload["request_id"] = self.request_index
        self.websocket.send(_pack(payload))
        response = _unpack(self.websocket.recv())
        if not isinstance(response, dict):
            raise TypeError(
                "Policy server returned {}, expected dict".format(type(response).__name__)
            )
        if response.get("request_id") not in (None, self.request_index):
            raise RuntimeError(
                "Mismatched response request_id: expected {}, got {}".format(
                    self.request_index, response.get("request_id")
                )
            )
        return response

    def _wait_for_server(self, timeout):
        logging.info("Waiting for policy server at %s", self.uri)
        started = time.time()
        # A malformed SOCKS proxy must not intercept a localhost websocket.
        for key in (
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
        ):
            os.environ.pop(key, None)

        while True:
            if time.time() - started > timeout:
                raise TimeoutError(
                    "Failed to connect to policy server within {} seconds".format(timeout)
                )
            try:
                websocket = _connect(self.uri)
                handshake = _unpack(websocket.recv())
                _raise_for_error(handshake)
                if handshake.get("type") != "handshake":
                    raise RuntimeError("Expected server handshake, got {!r}".format(handshake))
                metadata = handshake.get("metadata", {})
                if metadata.get("protocol_version") != 1:
                    raise RuntimeError(
                        "Unsupported policy protocol version: {}".format(
                            metadata.get("protocol_version")
                        )
                    )
                logging.info("Connected to policy server metadata=%s", metadata)
                return websocket, metadata
            except (ConnectionRefusedError, OSError):
                time.sleep(2.0)


def _connect(uri):
    try:
        import websockets.sync.client
    except (ImportError, ModuleNotFoundError) as exc:
        raise ModuleNotFoundError(
            "Install CALVIN evaluation dependencies from "
            "calvin/evaluation/remote_policy/requirements.txt"
        ) from exc
    return websockets.sync.client.connect(
        uri,
        compression=None,
        max_size=None,
        open_timeout=150,
    )


def _pack(value):
    try:
        import msgpack
        import msgpack_numpy
    except (ImportError, ModuleNotFoundError) as exc:
        raise ModuleNotFoundError(
            "Install CALVIN evaluation dependencies from "
            "calvin/evaluation/remote_policy/requirements.txt"
        ) from exc
    msgpack_numpy.patch()
    return msgpack.packb(value, use_bin_type=True)


def _unpack(value):
    if isinstance(value, str):
        return json.loads(value)
    try:
        import msgpack
        import msgpack_numpy
    except (ImportError, ModuleNotFoundError) as exc:
        raise ModuleNotFoundError(
            "Install CALVIN evaluation dependencies from "
            "calvin/evaluation/remote_policy/requirements.txt"
        ) from exc
    msgpack_numpy.patch()
    return msgpack.unpackb(value, raw=False)


def _raise_for_error(response):
    if "error" in response:
        raise RuntimeError(
            "Policy server error {}: {}".format(
                response["error"], response.get("message", "")
            )
        )
