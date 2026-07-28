from __future__ import annotations

import argparse

from core.http_transport import serve
from core.mask_codec import decode_rgb
from qwen_service.backend import MockQwenBackend, QwenBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--backend", choices=("qwen", "mock"), default="qwen")
    parser.add_argument("--model-path", default="weights/Qwen3-VL-4B-Instruct")
    args = parser.parse_args()
    backend = (
        QwenBackend(args.model_path) if args.backend == "qwen" else MockQwenBackend()
    )

    def locate(payload: dict) -> dict:
        rgb = decode_rgb(payload["image_png"])
        result = backend.locate(payload["instruction"], rgb)
        result["frame_id"] = int(payload["frame_id"])
        return result

    serve(args.host, args.port, {"/locate": locate})


if __name__ == "__main__":
    main()
