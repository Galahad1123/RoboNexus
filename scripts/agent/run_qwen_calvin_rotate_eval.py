#!/usr/bin/env python3
"""Run single-skill Qwen-Agent rotate recovery trials from official CALVIN states."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
ROTATE_SKILLS = {
    "rotate_red_block_left",
    "rotate_red_block_right",
    "rotate_blue_block_left",
    "rotate_blue_block_right",
    "rotate_pink_block_left",
    "rotate_pink_block_right",
}


def build_manifest(source, selected_skills, limit):
    trials = []
    for sequence in source.get("sequences", ()):
        for position, skill_id in enumerate(sequence.get("skills", ()), 1):
            if skill_id not in selected_skills:
                continue
            trials.append(
                {
                    "index": len(trials),
                    "initial_condition": sequence["initial_condition"],
                    "skills": [skill_id],
                    "source_sequence_index": sequence.get("index"),
                    "source_skill_position": position,
                }
            )
            if limit is not None and len(trials) >= limit:
                break
        if limit is not None and len(trials) >= limit:
            break
    return {
        "benchmark": source.get("benchmark", "CALVIN-D"),
        "sequence_length": 1,
        "count": len(trials),
        "source_manifest": source.get("source_manifest"),
        "sequences": trials,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--skill",
        action="append",
        choices=sorted(ROTATE_SKILLS),
        help="repeat to test a subset; defaults to all six rotate skills",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--runtime-port", type=int, default=8121)
    parser.add_argument("--runtime-env", default="robonexus-calvin")
    parser.add_argument("--environment-step-limit", type=int, default=700)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-agent-rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = set(args.skill or ROTATE_SKILLS)
    manifest = build_manifest(source, selected, args.limit)
    if not manifest["sequences"]:
        parser.error("source manifest contains no selected rotate skills")

    run_dir = args.output_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    rotate_manifest = run_dir / "rotate_single_skill_manifest.json"
    rotate_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(WORKSPACE / "scripts/agent/run_qwen_calvin_batch.py"),
        "--manifest",
        str(rotate_manifest),
        "--output-dir",
        str(args.output_dir),
        "--runtime-port",
        str(args.runtime_port),
        "--runtime-env",
        args.runtime_env,
        "--environment-step-limit",
        str(args.environment_step_limit),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-agent-rounds",
        str(args.max_agent_rounds),
        "--seed",
        str(args.seed),
        "--approach",
    ]
    if args.model_path is not None:
        command.extend(["--model-path", str(args.model_path)])
    print(" ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=WORKSPACE, check=True)


if __name__ == "__main__":
    main()
