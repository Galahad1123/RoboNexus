#!/usr/bin/env python3
"""Materialize the deterministic official CALVIN length-5 evaluation sequence list."""

import argparse
import json
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
CALVIN_MODELS = WORKSPACE / "calvin/calvin_models"
sys.path.insert(0, str(CALVIN_MODELS))

from calvin_agent.evaluation.multistep_sequences import get_sequences  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.count <= 1000:
        parser.error("--count must be between 1 and 1000")
    sequences = [
        {"index": index, "initial_condition": state, "skills": list(skills)}
        for index, (state, skills) in enumerate(
            get_sequences(args.count, num_workers=args.workers)
        )
    ]
    document = {
        "protocol": "official_calvin_long_horizon",
        "sequence_length": 5,
        "count": len(sequences),
        "sequences": sequences,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
