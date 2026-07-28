from __future__ import annotations

import argparse

from core.http_transport import serve
from core.mask_codec import decode_rgb, encode_mask
from segmentation_service.backend import MockSamBackend, SamBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--backend", choices=("sam", "mock"), default="sam")
    parser.add_argument(
        "--checkpoint",
        default="RoboNexus-Agent/third_party/segment-anything/checkpoints/sam_vit_b_01ec64.pth",
    )
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    backend = (
        SamBackend(args.checkpoint, args.model_type, args.device)
        if args.backend == "sam"
        else MockSamBackend()
    )

    def segment(payload: dict) -> dict:
        results = backend.segment(decode_rgb(payload["image_png"]), payload["prompts"])
        return {
            "frame_id": int(payload["frame_id"]),
            "masks": [
                {"mask_png": encode_mask(mask), "confidence": score}
                for mask, score in results
            ],
        }

    serve(args.host, args.port, {"/segment": segment})


if __name__ == "__main__":
    main()
