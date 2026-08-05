"""Reusable clients for checkpoint-backed DiT policy endpoints."""

import json
import os
from types import MappingProxyType


PROTOCOL_VERSION = 2
POLICY_KIND = "multitask_dit_policy"
CONTRACT_FIELDS = (
    "protocol_version",
    "policy",
    "image_keys",
    "image_shapes",
    "state_key",
    "state_dim",
    "action_dim",
    "n_obs_steps",
    "model_action_chunk_size",
    "horizon",
)


class PolicyConnectionError(RuntimeError):
    pass


class HandshakeValidationError(PolicyConnectionError):
    pass


class PolicyEndpointClient:
    """Small Python-3.8-compatible client for policy protocol v2."""

    def __init__(self, policy, connect_timeout=30.0):
        self.policy = policy
        self.request_index = 0
        self.websocket = self._connect(policy.endpoint, connect_timeout)
        try:
            handshake = self._unpack(self.websocket.recv())
            if not isinstance(handshake, dict) or handshake.get("type") != "handshake":
                raise HandshakeValidationError(
                    "{} returned an invalid handshake: {!r}".format(
                        policy.endpoint, handshake
                    )
                )
            self.metadata = handshake.get("metadata")
        except Exception:
            self.close()
            raise

    def request(self, payload):
        self.request_index += 1
        request = dict(payload)
        request["request_id"] = self.request_index
        self.websocket.send(self._pack(request))
        response = self._unpack(self.websocket.recv())
        if not isinstance(response, dict):
            raise PolicyConnectionError(
                "{} returned {}, expected mapping".format(
                    self.policy.endpoint, type(response).__name__
                )
            )
        if response.get("request_id") not in (None, self.request_index):
            raise PolicyConnectionError(
                "{} returned mismatched request_id {}".format(
                    self.policy.endpoint, response.get("request_id")
                )
            )
        if "error" in response:
            raise PolicyConnectionError(
                "{} returned {}: {}".format(
                    self.policy.endpoint,
                    response["error"],
                    response.get("message", ""),
                )
            )
        return response

    def health(self):
        return self.request({"type": "health"})

    def reset(self, episode_id, task, replan_interval, episode_seed):
        return self.request(
            {
                "type": "reset",
                "episode_id": str(episode_id),
                "task": str(task),
                "replan_interval": int(replan_interval),
                "episode_seed": int(episode_seed),
            }
        )

    def step(self, episode_id, observation, step):
        return self.request(
            {
                "type": "step",
                "episode_id": str(episode_id),
                "observation": observation,
                "step": int(step),
            }
        )

    def close(self):
        websocket = getattr(self, "websocket", None)
        if websocket is not None:
            try:
                websocket.close()
            finally:
                self.websocket = None

    @staticmethod
    def _connect(endpoint, timeout):
        try:
            import websockets.sync.client
        except (ImportError, ModuleNotFoundError) as exc:
            raise ModuleNotFoundError(
                "Policy transport requires websockets>=13,<14"
            ) from exc
        for key in (
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
        ):
            os.environ.pop(key, None)
        try:
            return websockets.sync.client.connect(
                endpoint,
                compression=None,
                max_size=None,
                open_timeout=timeout,
            )
        except Exception as exc:
            raise PolicyConnectionError(
                "Failed to connect to {}: {}".format(endpoint, exc)
            ) from exc

    @staticmethod
    def _pack(value):
        try:
            import msgpack
            import msgpack_numpy
        except (ImportError, ModuleNotFoundError) as exc:
            raise ModuleNotFoundError(
                "Policy transport requires msgpack and msgpack-numpy"
            ) from exc
        msgpack_numpy.patch()
        return msgpack.packb(value, use_bin_type=True)

    @staticmethod
    def _unpack(value):
        if isinstance(value, str):
            return json.loads(value)
        try:
            import msgpack
            import msgpack_numpy
        except (ImportError, ModuleNotFoundError) as exc:
            raise ModuleNotFoundError(
                "Policy transport requires msgpack and msgpack-numpy"
            ) from exc
        msgpack_numpy.patch()
        return msgpack.unpackb(value, raw=False)


class PolicyClientPool:
    """Lazily creates one reusable connection for each enabled policy."""

    def __init__(self, catalog, client_factory=None, connect_timeout=30.0):
        self.catalog = catalog
        self.connect_timeout = connect_timeout
        self.client_factory = client_factory or self._default_client_factory
        self._clients = {}
        self._reference_contract = None

    @property
    def connected_policy_ids(self):
        return tuple(sorted(self._clients))

    @property
    def reference_contract(self):
        if self._reference_contract is None:
            return None
        return MappingProxyType(dict(self._reference_contract))

    def get(self, policy_id):
        if policy_id in self._clients:
            return self._clients[policy_id]
        policy = self.catalog.get(policy_id)
        if not policy.enabled:
            raise PolicyConnectionError("Policy {!r} is disabled".format(policy_id))
        client = None
        try:
            client = self.client_factory(policy)
            contract = _validate_handshake(
                policy_id, getattr(client, "metadata", None)
            )
            if self._reference_contract is None:
                self._reference_contract = contract
            else:
                _validate_compatible(policy_id, self._reference_contract, contract)
            self._clients[policy_id] = client
            return client
        except Exception:
            if client is not None:
                client.close()
            raise

    def connect_all(self, policy_ids=None):
        selected = tuple(policy_ids or self.catalog.policy_ids())
        previously_connected = set(self._clients)
        try:
            return {policy_id: self.get(policy_id) for policy_id in selected}
        except Exception:
            for policy_id in set(self._clients) - previously_connected:
                self.close(policy_id)
            raise

    def health_check_all(self, policy_ids=None):
        clients = self.connect_all(policy_ids)
        results = {}
        for policy_id, client in clients.items():
            response = client.health()
            if response.get("type") != "health" or response.get("ok") is not True:
                raise PolicyConnectionError(
                    "Policy {!r} failed health check: {!r}".format(policy_id, response)
                )
            results[policy_id] = response
        return results

    def close(self, policy_id):
        client = self._clients.pop(policy_id, None)
        if client is not None:
            client.close()
        if not self._clients:
            self._reference_contract = None

    def close_all(self):
        for policy_id in tuple(self._clients):
            self.close(policy_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close_all()

    def _default_client_factory(self, policy):
        return PolicyEndpointClient(policy, connect_timeout=self.connect_timeout)


def _validate_handshake(policy_id, metadata):
    errors = []
    if not isinstance(metadata, dict):
        raise HandshakeValidationError(
            "Policy {!r} handshake metadata must be a mapping".format(policy_id)
        )
    missing = [field for field in CONTRACT_FIELDS if field not in metadata]
    if missing:
        errors.append("missing fields {}".format(missing))
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        errors.append("protocol_version must be {}".format(PROTOCOL_VERSION))
    if metadata.get("policy") != POLICY_KIND:
        errors.append("policy must be {!r}".format(POLICY_KIND))
    image_keys = metadata.get("image_keys")
    image_shapes = metadata.get("image_shapes")
    if not isinstance(image_keys, list) or not image_keys:
        errors.append("image_keys must be a non-empty list")
    if not isinstance(image_shapes, dict):
        errors.append("image_shapes must be a mapping")
    elif isinstance(image_keys, list):
        if set(image_shapes) != set(image_keys):
            errors.append("image_shapes keys must exactly match image_keys")
        for key in image_keys:
            shape = image_shapes.get(key)
            if (
                not isinstance(shape, (list, tuple))
                or len(shape) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 1
                    for value in shape
                )
            ):
                errors.append(
                    "image_shapes.{!s} must contain three positive integers".format(
                        key
                    )
                )
    if not isinstance(metadata.get("state_key"), str) or not metadata.get("state_key"):
        errors.append("state_key must be a non-empty string")
    for field in (
        "state_dim",
        "action_dim",
        "n_obs_steps",
        "model_action_chunk_size",
        "horizon",
    ):
        value = metadata.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            errors.append("{} must be a positive integer".format(field))
    if errors:
        raise HandshakeValidationError(
            "Policy {!r} has invalid handshake:\n- {}".format(
                policy_id, "\n- ".join(errors)
            )
        )
    return {field: _freeze(metadata[field]) for field in CONTRACT_FIELDS}


def _validate_compatible(policy_id, expected, actual):
    differences = []
    for field in CONTRACT_FIELDS:
        if actual[field] != expected[field]:
            differences.append(
                "{} expected {!r}, got {!r}".format(
                    field, expected[field], actual[field]
                )
            )
    if differences:
        raise HandshakeValidationError(
            "Policy {!r} handshake is incompatible:\n- {}".format(
                policy_id, "\n- ".join(differences)
            )
        )


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value
