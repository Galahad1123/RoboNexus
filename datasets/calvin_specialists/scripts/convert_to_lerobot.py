#!/usr/bin/env python3
"""Dataset-selecting wrapper around the shared CALVIN LeRobot converter."""

import argparse
import importlib.util
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[3]
CONVERTER = WORKSPACE / "datasets/calvin_d_10/scripts/convert_to_lerobot.py"


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", required=True)
    known, remaining = parser.parse_known_args()
    manifest = WORKSPACE / "datasets" / known.dataset / "manifests/segments.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not any(value == "--output" or value.startswith("--output=") for value in remaining):
        raise ValueError("Specialist conversion requires an explicit --output")
    sys.argv = [sys.argv[0], "--manifest", str(manifest)] + remaining
    spec = importlib.util.spec_from_file_location("calvin_lerobot_converter", CONVERTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
