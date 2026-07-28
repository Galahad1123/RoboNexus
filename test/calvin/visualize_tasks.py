#!/usr/bin/env python3
"""Interactively browse or export one representative clip per CALVIN task."""

import argparse
from collections import defaultdict
from pathlib import Path
import re
from typing import Dict, List, Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "calvin" / "dataset" / "task_D_D"


def load_annotations(dataset: Path, split: str) -> Dict[str, List[dict]]:
    annotation_file = dataset / split / "lang_annotations" / "auto_lang_ann.npy"
    data = np.load(annotation_file, allow_pickle=True).item()
    grouped = defaultdict(list)
    for task, text, interval in zip(
        data["language"]["task"], data["language"]["ann"], data["info"]["indx"]
    ):
        start, end = map(int, interval)
        grouped[str(task)].append(
            {"task": str(task), "text": str(text), "start": start, "end": end, "length": end - start + 1}
        )
    return dict(grouped)


def representative_clip(clips: Sequence[dict], selection: str, seed: int) -> dict:
    if selection == "first":
        return clips[0]
    if selection == "random":
        return clips[int(np.random.RandomState(seed).randint(len(clips)))]
    lengths = np.asarray([clip["length"] for clip in clips])
    median = np.median(lengths)
    return clips[int(np.argmin(np.abs(lengths - median)))]


def wrap_text(text: str, max_chars: int = 68) -> List[str]:
    words = text.split()
    lines, current = [], []
    for word in words:
        if current and len(" ".join(current + [word])) > max_chars:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def add_panel(frame: np.ndarray, clip: dict, index: int, total: int, paused: bool) -> np.ndarray:
    height, width = frame.shape[:2]
    panel_height = 118
    canvas = np.zeros((height + panel_height, width, 3), dtype=np.uint8)
    canvas[:height] = frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        f"[{index + 1}/{total}] {clip['task']}",
        (12, height + 27),
        font,
        0.64,
        (80, 220, 255),
        2,
        cv2.LINE_AA,
    )
    for line_index, line in enumerate(wrap_text(clip["text"])[:2]):
        cv2.putText(
            canvas,
            line,
            (12, height + 53 + line_index * 21),
            font,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    help_text = "SPACE pause | R replay | N next | P previous | Q quit"
    if paused:
        help_text = "PAUSED | " + help_text
    cv2.putText(canvas, help_text, (12, height + 108), font, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    return canvas


def compose_frame(
    episode: np.lib.npyio.NpzFile, clip: dict, frame_number: int, total_frames: int, paused: bool
) -> np.ndarray:
    static = cv2.cvtColor(episode["rgb_static"], cv2.COLOR_RGB2BGR)
    gripper = cv2.cvtColor(episode["rgb_gripper"], cv2.COLOR_RGB2BGR)
    display_height = static.shape[0]
    gripper = cv2.resize(gripper, (display_height, display_height), interpolation=cv2.INTER_NEAREST)
    frame = np.hstack((static, gripper))
    cv2.putText(frame, "static", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "gripper",
        (static.shape[1] + 8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    progress = int(frame.shape[1] * (frame_number + 1) / total_frames)
    cv2.rectangle(frame, (0, frame.shape[0] - 4), (progress, frame.shape[0] - 1), (80, 220, 255), -1)
    return frame


def frame_path(split_dir: Path, frame_id: int) -> Path:
    return split_dir / f"episode_{frame_id:07d}.npz"


def load_display_frame(
    split_dir: Path, clip: dict, offset: int, task_index: int, task_total: int, paused: bool
) -> np.ndarray:
    path = frame_path(split_dir, clip["start"] + offset)
    with np.load(path) as episode:
        frame = compose_frame(episode, clip, offset, clip["length"], paused)
    return add_panel(frame, clip, task_index, task_total, paused)


def browse(split_dir: Path, clips: Sequence[dict], fps: float, scale: float) -> None:
    task_index, offset, paused = 0, 0, False
    delay = max(1, round(1000 / fps))
    window = "CALVIN task browser"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while True:
        clip = clips[task_index]
        frame = load_display_frame(split_dir, clip, offset, task_index, len(clips), paused)
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(window, frame)
        key = cv2.waitKey(0 if paused else delay) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            offset = 0
        elif key in (ord("n"), 83):
            task_index = (task_index + 1) % len(clips)
            offset = 0
        elif key in (ord("p"), 81):
            task_index = (task_index - 1) % len(clips)
            offset = 0
        elif not paused:
            offset += 1
            if offset >= clip["length"]:
                task_index = (task_index + 1) % len(clips)
                offset = 0
    cv2.destroyAllWindows()


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def export_clips(split_dir: Path, clips: Sequence[dict], export_dir: Path, fps: float, scale: float) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    for task_index, clip in enumerate(clips):
        first = load_display_frame(split_dir, clip, 0, task_index, len(clips), False)
        if scale != 1.0:
            first = cv2.resize(first, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        height, width = first.shape[:2]
        output = export_dir / f"{task_index + 1:02d}_{safe_name(clip['task'])}.mp4"
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer for {output}")
        for offset in range(clip["length"]):
            frame = load_display_frame(split_dir, clip, offset, task_index, len(clips), False)
            if scale != 1.0:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
            writer.write(frame)
        writer.release()
        print(f"[{task_index + 1:02d}/{len(clips):02d}] {output} ({clip['length']} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("training", "validation"), default="validation")
    parser.add_argument("--task", action="append", help="task to show; repeat it, or omit to show all tasks")
    parser.add_argument("--selection", choices=("median", "first", "random"), default="median")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--export-dir", type=Path, help="export MP4 clips instead of opening a GUI")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    annotations = load_annotations(dataset, args.split)
    requested = args.task or sorted(annotations)
    unknown = sorted(set(requested).difference(annotations))
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}; available tasks: {sorted(annotations)}")
    clips = [representative_clip(annotations[task], args.selection, args.seed + i) for i, task in enumerate(requested)]
    split_dir = dataset / args.split
    print(f"Loaded {len(clips)} task clips from {split_dir}")

    if args.export_dir:
        export_clips(split_dir, clips, args.export_dir.resolve(), args.fps, args.scale)
    else:
        browse(split_dir, clips, args.fps, args.scale)


if __name__ == "__main__":
    main()
