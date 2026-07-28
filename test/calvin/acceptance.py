#!/usr/bin/env python3
"""Independent smoke/acceptance checks for the CALVIN simulator.

Each simulator mode is run in a child process because a broken EGL setup can
terminate the interpreter instead of raising a Python exception.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "calvin" / "dataset" / "calvin_debug_dataset"
REQUIRED_KEYS = {
    "actions",
    "rel_actions",
    "robot_obs",
    "scene_obs",
    "rgb_static",
    "rgb_gripper",
    "depth_static",
    "depth_gripper",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _check_array(name: str, value: np.ndarray, shape: Iterable[int]) -> None:
    if value.shape != tuple(shape):
        raise AssertionError(f"{name}: expected shape {tuple(shape)}, got {value.shape}")
    if not np.isfinite(value).all():
        raise AssertionError(f"{name}: contains NaN or Inf")


def inspect_dataset(dataset_root: Path, split: str) -> Dict[str, Any]:
    split_dir = dataset_root / split
    config = split_dir / ".hydra" / "merged_config.yaml"
    ranges_file = split_dir / "ep_start_end_ids.npy"
    files = sorted(split_dir.glob("episode_*.npz"))
    if not config.is_file() or not ranges_file.is_file() or not files:
        raise FileNotFoundError(f"incomplete CALVIN split: {split_dir}")

    ranges = np.load(ranges_file)
    with np.load(files[0]) as frame:
        missing = REQUIRED_KEYS.difference(frame.files)
        if missing:
            raise AssertionError(f"{files[0]} is missing keys: {sorted(missing)}")
        _check_array("robot_obs", frame["robot_obs"], (15,))
        _check_array("scene_obs", frame["scene_obs"], (24,))
        _check_array("rel_actions", frame["rel_actions"], (7,))
        shapes = {key: list(frame[key].shape) for key in sorted(REQUIRED_KEYS)}

    return {
        "split": split,
        "frames": len(files),
        "first_frame": files[0].name,
        "episode_ranges": int(len(ranges)),
        "shapes": shapes,
    }


def make_env(dataset_split: Path, mode: str):
    import hydra
    from omegaconf import OmegaConf

    config = OmegaConf.load(dataset_split / ".hydra" / "merged_config.yaml")
    # The debug recording config includes an optional TACTO process. It is not
    # part of CALVIN's standard static/gripper camera observation and, without
    # DISPLAY, its failed worker can block forever waiting on a multiprocessing
    # queue. TACTO deserves a separate hardware-specific acceptance stage.
    if "tactile" in config.cameras:
        del config.cameras["tactile"]
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    return hydra.utils.instantiate(
        config.env,
        show_gui=mode == "gui",
        use_egl=mode == "egl",
        use_vr=False,
        use_scene_info=True,
    )


def validate_obs(obs: Dict[str, Any]) -> None:
    _check_array("robot_obs", obs["robot_obs"], (15,))
    _check_array("scene_obs", obs["scene_obs"], (24,))
    for key, shape in (("rgb_static", (200, 200, 3)), ("rgb_gripper", (84, 84, 3))):
        image = obs["rgb_obs"][key]
        _check_array(key, image, shape)
        if image.dtype != np.uint8:
            raise AssertionError(f"{key}: expected uint8, got {image.dtype}")
    for key, shape in (("depth_static", (200, 200)), ("depth_gripper", (84, 84))):
        _check_array(key, obs["depth_obs"][key], shape)


def run_simulator_check(dataset_root: Path, split: str, mode: str, steps: int) -> Dict[str, Any]:
    split_dir = dataset_root / split
    env = None
    started = time.monotonic()
    try:
        env = make_env(split_dir, mode)
        obs = env.reset()
        validate_obs(obs)

        render_sums = []
        zero_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        for _ in range(steps):
            obs, reward, done, info = env.step(zero_action)
            validate_obs(obs)
            render_sums.append(int(obs["rgb_obs"]["rgb_static"].sum()))
            if reward != 0 or done is not False or "robot_info" not in info:
                raise AssertionError("unexpected CALVIN step() return contract")

        frame_path = sorted(split_dir.glob("episode_*.npz"))[0]
        with np.load(frame_path) as frame:
            restored = env.reset(robot_obs=frame["robot_obs"], scene_obs=frame["scene_obs"])
            validate_obs(restored)
            robot_error = float(np.max(np.abs(restored["robot_obs"] - frame["robot_obs"])))
            scene_error = float(np.max(np.abs(restored["scene_obs"] - frame["scene_obs"])))
            action = frame["rel_actions"].copy()

        stepped, _, _, _ = env.step(action)
        validate_obs(stepped)
        return {
            "mode": mode,
            "steps": steps,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "camera_checksums": render_sums,
            "camera_frames_nonzero": all(value > 0 for value in render_sums),
            "restored_frame": frame_path.name,
            "restore_max_abs_error": {"robot_obs": robot_error, "scene_obs": scene_error},
            "dataset_action_executed": action.tolist(),
        }
    finally:
        if env is not None:
            env.close()
            # CALVIN's destructor otherwise disconnects the same Bullet client
            # a second time and prints a misleading ignored exception.
            env.ownsPhysicsClient = False


def run_child(args: argparse.Namespace, mode: str) -> Dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        mode,
        "--dataset",
        str(args.dataset),
        "--split",
        args.split,
        "--steps",
        str(args.steps),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired as exc:
        return {
            "mode": mode,
            "passed": False,
            "returncode": None,
            "error": f"worker timed out after {args.timeout}s",
            "stdout": (exc.stdout or b"").decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or ""),
            "stderr": (exc.stderr or b"").decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or ""),
        }
    result = {
        "mode": mode,
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode == 0:
        lines = [line for line in completed.stdout.splitlines() if line.startswith("CALVIN_RESULT=")]
        if not lines:
            result["passed"] = False
            result["error"] = "worker returned no structured result"
        else:
            result["details"] = json.loads(lines[-1].split("=", 1)[1])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("training", "validation"), default="validation")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--gui", action="store_true", help="also launch the GUI check (requires DISPLAY)")
    parser.add_argument("--direct", action="store_true", help="also check CPU/direct rendering")
    parser.add_argument("--report", type=Path, help="write the complete JSON report here")
    parser.add_argument("--worker", choices=("egl", "gui", "direct"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()

    if args.worker:
        details = run_simulator_check(args.dataset, args.split, args.worker, args.steps)
        print("CALVIN_RESULT=" + json.dumps(details, default=_jsonable, sort_keys=True))
        return 0

    report: Dict[str, Any] = {
        "dataset": str(args.dataset),
        "dataset_check": inspect_dataset(args.dataset, args.split),
        "checks": [],
    }
    modes = ["egl"]
    if args.direct:
        modes.append("direct")
    if args.gui:
        modes.append("gui")
    for mode in modes:
        if mode == "gui" and not os.environ.get("DISPLAY"):
            report["checks"].append(
                {"mode": "gui", "passed": False, "skipped": True, "error": "DISPLAY is not set"}
            )
        else:
            report["checks"].append(run_child(args, mode))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, default=_jsonable, indent=2) + "\n")
    print(json.dumps(report, default=_jsonable, indent=2))
    return 0 if all(check["passed"] or check.get("skipped") for check in report["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
