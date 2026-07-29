"""Wire protocol helpers shared by policy-server entry points."""

from __future__ import annotations

import json
from typing import Any

PROTOCOL_VERSION = 2


def pack_message(value: Any) -> bytes:
    try:
        import msgpack
        import msgpack_numpy
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Policy serving requires websockets, msgpack and msgpack-numpy. Run `uv sync`."
        ) from exc
    msgpack_numpy.patch()
    return msgpack.packb(value, use_bin_type=True)


def unpack_message(value: bytes | str) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    try:
        import msgpack
        import msgpack_numpy
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Policy serving requires websockets, msgpack and msgpack-numpy. Run `uv sync`."
        ) from exc
    msgpack_numpy.patch()
    return msgpack.unpackb(value, raw=False)
