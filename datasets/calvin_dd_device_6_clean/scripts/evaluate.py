#!/usr/bin/env python3
"""Evaluate the clean CALVIN-D Device-6 policy on task_D_D validation."""

import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_SUITE = WORKSPACE / "datasets/calvin_dd_device_6_clean/configs/evaluation.yaml"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
if "--suite" not in sys.argv:
    sys.argv.extend(["--suite", str(DEFAULT_SUITE)])

from calvin.evaluation.remote_policy.evaluate import parse_args, run  # noqa: E402


if __name__ == "__main__":
    run(parse_args())
