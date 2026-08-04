#!/usr/bin/env python3
"""Evaluate the direction-conditioned CALVIN Rotate-2 specialist suite."""

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[2]
DEFAULT_SUITE = (
    WORKSPACE / "datasets/calvin_rotate_2_release/configs/evaluation.yaml"
)

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
if "--suite" not in sys.argv:
    sys.argv.extend(["--suite", str(DEFAULT_SUITE)])

from calvin.evaluation.remote_policy.evaluate import parse_args, run  # noqa: E402


if __name__ == "__main__":
    run(parse_args())
