#!/usr/bin/env python3
"""Browse or export synchronized A/B/C/D demonstrations for all CALVIN tasks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "calvin" / "dataset" / "task_ABC_D"
ENVIRONMENTS = ("A", "B", "C", "D")
ENV_COLORS = {
    "A": (90, 220, 255),
    "B": (110, 235, 120),
    "C": (255, 180, 90),
    "D": (210, 120, 255),
}


def load_scene_ranges(dataset: Path) -> Dict[str, Tuple[int, int]]:
    path = dataset / "training" / "scene_info.npy"
    raw = np.load(path, allow_pickle=True).item()
    ranges = {}
    for key, value in raw.items():
        match = re.search(r"(?:scene|env)_([A-C])$", str(key))
        if match:
            ranges[match.group(1)] = tuple(map(int, value))
    missing = sorted(set(ENVIRONMENTS[:3]) - set(ranges))
    if missing:
        raise ValueError(f"{path} does not define scene ranges for {missing}: {raw}")
    return ranges


def environment_for_interval(start: int, end: int, ranges: Mapping[str, Tuple[int, int]]) -> str:
    matches = [name for name, (lower, upper) in ranges.items() if lower <= start and end <= upper]
    if len(matches) != 1:
        raise ValueError(f"annotation interval [{start}, {end}] does not belong to exactly one scene: {ranges}")
    return matches[0]


def load_annotations(dataset: Path) -> Dict[str, Dict[str, List[dict]]]:
    ranges = load_scene_ranges(dataset)
    grouped: Dict[str, Dict[str, List[dict]]] = {
        env: defaultdict(list) for env in ENVIRONMENTS
    }
    for split in ("training", "validation"):
        path = dataset / split / "lang_annotations" / "auto_lang_ann.npy"
        data = np.load(path, allow_pickle=True).item()
        for task, text, interval in zip(
            data["language"]["task"], data["language"]["ann"], data["info"]["indx"]
        ):
            start, end = map(int, interval)
            environment = "D" if split == "validation" else environment_for_interval(start, end, ranges)
            grouped[environment][str(task)].append(
                {
                    "environment": environment,
                    "split": split,
                    "task": str(task),
                    "text": str(text),
                    "start": start,
                    "end": end,
                    "length": end - start + 1,
                }
            )
    return {env: dict(tasks) for env, tasks in grouped.items()}


def representative_clip(clips: Sequence[dict], selection: str, seed: int) -> dict:
    if selection == "first":
        return clips[0]
    if selection == "random":
        return clips[int(np.random.RandomState(seed).randint(len(clips)))]
    lengths = np.asarray([clip["length"] for clip in clips])
    return clips[int(np.argmin(np.abs(lengths - np.median(lengths))))]


def frame_path(dataset: Path, clip: Mapping[str, object], frame_id: int) -> Path:
    return dataset / str(clip["split"]) / f"episode_{frame_id:07d}.npz"


def tile_frame(dataset: Path, clip: Mapping[str, object], normalized_progress: float) -> np.ndarray:
    offset = min(int(round(normalized_progress * (int(clip["length"]) - 1))), int(clip["length"]) - 1)
    path = frame_path(dataset, clip, int(clip["start"]) + offset)
    with np.load(path) as episode:
        static = cv2.cvtColor(episode["rgb_static"], cv2.COLOR_RGB2BGR)
        gripper = cv2.cvtColor(episode["rgb_gripper"], cv2.COLOR_RGB2BGR)
    gripper = cv2.resize(gripper, (static.shape[0], static.shape[0]), interpolation=cv2.INTER_NEAREST)
    tile = np.hstack((static, gripper))
    environment = str(clip["environment"])
    color = ENV_COLORS[environment]
    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), color, 3)
    cv2.putText(tile, f"Environment {environment}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(tile, "static", (10, tile.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(tile, "gripper", (static.shape[1] + 10, tile.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
    progress_width = int((tile.shape[1] - 6) * normalized_progress)
    cv2.rectangle(tile, (3, tile.shape[0] - 5), (3 + progress_width, tile.shape[0] - 3), color, -1)
    return tile


def wrap_text(text: str, max_chars: int) -> List[str]:
    words, lines, current = text.split(), [], []
    for word in words:
        if current and len(" ".join(current + [word])) > max_chars:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def compose_frame(
    dataset: Path,
    clips: Mapping[str, dict],
    normalized_progress: float,
    task_index: int,
    task_total: int,
    paused: bool,
) -> np.ndarray:
    tiles = {env: tile_frame(dataset, clips[env], normalized_progress) for env in ENVIRONMENTS}
    grid = np.vstack((np.hstack((tiles["A"], tiles["B"])), np.hstack((tiles["C"], tiles["D"]))))
    panel_height = 112
    canvas = np.zeros((grid.shape[0] + panel_height, grid.shape[1], 3), dtype=np.uint8)
    canvas[: grid.shape[0]] = grid
    task = clips["A"]["task"]
    cv2.putText(canvas, f"[{task_index + 1}/{task_total}] {task}", (14, grid.shape[0] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (90, 220, 255), 2, cv2.LINE_AA)
    descriptions = " | ".join(f"{env}: {clips[env]['text']}" for env in ENVIRONMENTS)
    for line_index, line in enumerate(wrap_text(descriptions, 115)[:2]):
        cv2.putText(canvas, line, (14, grid.shape[0] + 53 + line_index * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 235, 235), 1, cv2.LINE_AA)
    help_text = "SPACE pause | R replay | N/right next | P/left previous | Q/Esc quit"
    if paused:
        help_text = "PAUSED | " + help_text
    cv2.putText(canvas, help_text, (14, grid.shape[0] + 103), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (165, 165, 165), 1, cv2.LINE_AA)
    return canvas


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def browse(dataset: Path, tasks: Sequence[str], selected: Mapping[str, Mapping[str, dict]], fps: float, scale: float) -> None:
    task_index, output_offset, paused = 0, 0, False
    delay = max(1, round(1000 / fps))
    window = "CALVIN ABC -> D task comparison"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while True:
        task = tasks[task_index]
        clips = selected[task]
        duration = max(clip["length"] for clip in clips.values())
        progress = output_offset / max(1, duration - 1)
        frame = compose_frame(dataset, clips, progress, task_index, len(tasks), paused)
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(window, frame)
        key = cv2.waitKey(0 if paused else delay) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            output_offset = 0
        elif key in (ord("n"), 83):
            task_index = (task_index + 1) % len(tasks)
            output_offset = 0
        elif key in (ord("p"), 81):
            task_index = (task_index - 1) % len(tasks)
            output_offset = 0
        elif not paused:
            output_offset += 1
            if output_offset >= duration:
                task_index = (task_index + 1) % len(tasks)
                output_offset = 0
    cv2.destroyAllWindows()


def export_clips(
    dataset: Path,
    tasks: Sequence[str],
    selected: Mapping[str, Mapping[str, dict]],
    export_dir: Path,
    fps: float,
    scale: float,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": str(dataset), "selection": {}}
    for task_index, task in enumerate(tasks):
        clips = selected[task]
        duration = max(clip["length"] for clip in clips.values())
        first = compose_frame(dataset, clips, 0.0, task_index, len(tasks), False)
        if scale != 1.0:
            first = cv2.resize(first, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        height, width = first.shape[:2]
        output = export_dir / f"{task_index + 1:02d}_{safe_name(task)}.mp4"
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer for {output}")
        for offset in range(duration):
            progress = offset / max(1, duration - 1)
            frame = compose_frame(dataset, clips, progress, task_index, len(tasks), False)
            if scale != 1.0:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
            writer.write(frame)
        writer.release()
        manifest["selection"][task] = {
            env: {key: clip[key] for key in ("split", "text", "start", "end", "length")}
            for env, clip in clips.items()
        }
        print(f"[{task_index + 1:02d}/{len(tasks):02d}] {output} ({duration} synchronized frames)")
    (export_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--task", action="append", help="task to show; repeat it, or omit for all 34")
    parser.add_argument("--selection", choices=("median", "first", "random"), default="median")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--export-dir", type=Path, help="export one synchronized MP4 per task instead of opening a GUI")
    args = parser.parse_args()
    if args.fps <= 0 or args.scale <= 0:
        parser.error("--fps and --scale must be positive")

    dataset = args.dataset.resolve()
    annotations = load_annotations(dataset)
    common = set.intersection(*(set(annotations[env]) for env in ENVIRONMENTS))
    requested = args.task or sorted(common)
    unknown = sorted(set(requested) - common)
    if unknown:
        availability = {env: sorted(annotations[env]) for env in ENVIRONMENTS}
        raise ValueError(f"tasks are not available in every A/B/C/D environment: {unknown}; availability={availability}")
    selected = {
        task: {
            env: representative_clip(annotations[env][task], args.selection, args.seed + task_index * 4 + env_index)
            for env_index, env in enumerate(ENVIRONMENTS)
        }
        for task_index, task in enumerate(requested)
    }
    counts = {env: len(annotations[env]) for env in ENVIRONMENTS}
    print(f"Loaded {len(requested)} shared tasks from {dataset}; task counts by environment: {counts}")
    if args.export_dir:
        export_clips(dataset, requested, selected, args.export_dir.resolve(), args.fps, args.scale)
    else:
        browse(dataset, requested, selected, args.fps, args.scale)


if __name__ == "__main__":
    main()
