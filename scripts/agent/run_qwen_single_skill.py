#!/usr/bin/env python3
"""Run Qwen-Agent/Qwen3-VL against one remote robot goal session."""

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[1]
AGENT_ROOT = WORKSPACE / "RoboNexus-Agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from adapters.qwen_agent import run_single_skill_agent  # noqa: E402
from core.logging import recover_agent_result  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-url", default="http://127.0.0.1:8110")
    parser.add_argument(
        "--model-path",
        default=str(WORKSPACE / "weights/Qwen3-VL-4B-Instruct"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--expect-skill", action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--recover-output-dir",
        type=Path,
        help="rebuild qwen_agent_result.json from checkpoint/runtime artifacts without loading Qwen",
    )
    args = parser.parse_args()

    if args.recover_output_dir is not None:
        report = recover_agent_result(args.recover_output_dir)
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return 0 if report["effective_status"] == "succeeded" else 2

    report = run_single_skill_agent(
        args.runtime_url,
        args.model_path,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        output=args.output,
        expected_skill_ids=args.expect_skill,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report["effective_status"] == "succeeded" else 2


if __name__ == "__main__":
    sys.exit(main())
