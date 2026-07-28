"""Replay LeRobot CALVIN episodes through the perception pipeline."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from core.clients import CutieHttpClient, QwenHttpClient, SamHttpClient
from core.orchestrator import PerceptionPipeline
from core.schemas import FramePacket, public_dict


ROLE_COLORS = {"source": np.array([255, 40, 40]), "destination": np.array([40, 80, 255])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="datasets/calvin_d_10/data/calvin_d_10_validation_image",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=65)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--qwen-url", default="http://127.0.0.1:8101")
    parser.add_argument("--sam-url", default="http://127.0.0.1:8102")
    parser.add_argument("--cutie-url", default="http://127.0.0.1:8103")
    parser.add_argument("--output", default="/tmp/robonexus-replay")
    args = parser.parse_args()

    rows, instruction = load_episode(Path(args.dataset), args.episode)
    rows = rows[:: args.stride][: args.max_frames]
    if not rows:
        raise SystemExit(f"episode {args.episode} is empty")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pipeline = PerceptionPipeline(
        QwenHttpClient(args.qwen_url),
        SamHttpClient(args.sam_url),
        CutieHttpClient(args.cutie_url),
    )
    report = {"episode": args.episode, "instruction": instruction, "frames": []}
    try:
        for index, row in enumerate(rows):
            rgb = decode_image(row["observation.images.static"])
            packet = FramePacket(
                episode_id=f"calvin-{args.episode}",
                frame_id=int(row["frame_index"]),
                timestamp_ns=time.time_ns(),
                rgb=rgb,
            )
            result = (
                pipeline.start(instruction, packet)
                if index == 0
                else pipeline.process(packet)
            )
            report["frames"].append(public_dict(result))
            save_overlay(rgb, result.objects, output / f"{index:05d}.png")
    finally:
        pipeline.close(completed=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"wrote {len(rows)} frames and report to {output.resolve()}")


def load_episode(dataset: Path, episode: int) -> tuple[list[dict], str]:
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("CALVIN replay requires pyarrow (use the DiT venv)") from exc

    rows: list[dict] = []
    for parquet in sorted((dataset / "data").rglob("*.parquet")):
        table = pq.read_table(parquet)
        selected = table.filter(pc.equal(table["episode_index"], episode))
        rows.extend(selected.to_pylist())
    rows.sort(key=lambda item: item["frame_index"])
    if not rows:
        return [], ""
    task_index = int(rows[0]["task_index"])
    tasks = pq.read_table(dataset / "meta" / "tasks.parquet").to_pylist()
    task_row = next(item for item in tasks if int(item["task_index"]) == task_index)
    return rows, str(task_row["task"])


def decode_image(value: dict) -> np.ndarray:
    if value.get("bytes"):
        return np.asarray(Image.open(io.BytesIO(value["bytes"])).convert("RGB"))
    return np.asarray(Image.open(value["path"]).convert("RGB"))


def save_overlay(rgb: np.ndarray, objects: list, path: Path, alpha: float = 0.45) -> None:
    image = rgb.astype(np.float32)
    for obj in objects:
        color = ROLE_COLORS.get(obj.role.value, np.array([40, 220, 80]))
        mask = obj.mask.astype(bool)
        image[mask] = (1 - alpha) * image[mask] + alpha * color
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
