#!/usr/bin/env python3
"""Evaluate the clean ABC Manipulation-13 policy on audited task_D_D states."""
import sys
from pathlib import Path
WORKSPACE = Path(__file__).resolve().parents[3]
SUITE = WORKSPACE / "datasets/calvin_abc_manipulation_13_clean/configs/evaluation_task_D_D.yaml"
sys.path.insert(0, str(WORKSPACE))
if "--suite" not in sys.argv:
    sys.argv.extend(["--suite", str(SUITE)])
from calvin.evaluation.remote_policy.evaluate import parse_args, run  # noqa: E402
if __name__ == "__main__":
    run(parse_args())
