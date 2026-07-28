from __future__ import annotations

import argparse

from core.http_transport import serve
from core.mask_codec import decode_mask, decode_rgb, encode_mask
from tracking_service.backend import CutieBackend, MockCutieBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8103)
    parser.add_argument("--backend", choices=("cutie", "mock"), default="cutie")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-internal-size", type=int, default=480)
    args = parser.parse_args()
    backend = (
        CutieBackend(args.max_internal_size, args.device)
        if args.backend == "cutie"
        else MockCutieBackend()
    )

    def masks(payload: dict) -> dict[int, object]:
        return {
            int(object_id): decode_mask(mask)
            for object_id, mask in payload["masks"].items()
        }

    def response(payload: dict, result: dict) -> dict:
        return {
            "frame_id": int(payload["frame_id"]),
            "objects": [
                {
                    "object_id": object_id,
                    "mask_png": encode_mask(mask),
                    "confidence": confidence,
                }
                for object_id, (mask, confidence) in sorted(result.items())
            ],
        }

    def create(payload: dict) -> dict:
        backend.create_session(payload["session_id"])
        return {"session_id": payload["session_id"]}

    def initialize(payload: dict) -> dict:
        result = backend.initialize(
            payload["session_id"], decode_rgb(payload["image_png"]), masks(payload)
        )
        return response(payload, result)

    def track(payload: dict) -> dict:
        result = backend.track(payload["session_id"], decode_rgb(payload["image_png"]))
        return response(payload, result)

    def correct(payload: dict) -> dict:
        result = backend.correct(
            payload["session_id"], decode_rgb(payload["image_png"]), masks(payload)
        )
        return response(payload, result)

    def delete(payload: dict) -> dict:
        backend.delete_session(payload["session_id"])
        return {"session_id": payload["session_id"]}

    serve(
        args.host,
        args.port,
        {
            "/sessions/create": create,
            "/sessions/initialize": initialize,
            "/sessions/track": track,
            "/sessions/correct": correct,
            "/sessions/delete": delete,
        },
    )


if __name__ == "__main__":
    main()
