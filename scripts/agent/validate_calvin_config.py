#!/usr/bin/env python3
"""Validate CALVIN-D34 skill and policy configuration."""

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[1]
AGENT_ROOT = WORKSPACE / "RoboNexus-Agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from core.policies import load_policy_catalog  # noqa: E402
from core.skills import load_skill_registry  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skills",
        type=Path,
        default=AGENT_ROOT / "configs/calvin/skills_d34.yaml",
    )
    parser.add_argument(
        "--policies",
        type=Path,
        default=AGENT_ROOT / "configs/calvin/policies.yaml",
    )
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument(
        "--require-runnable",
        action="store_true",
        help="also require every policy to be enabled with a complete checkpoint",
    )
    args = parser.parse_args()

    registry = load_skill_registry(args.skills)
    policies = load_policy_catalog(
        args.policies,
        required_policy_ids=registry.policy_groups(),
    )
    if args.require_runnable:
        policies.validate_runtime(args.workspace, registry.policy_groups())

    result = {
        "ok": True,
        "skills": len(registry),
        "policy_groups": list(registry.policy_groups()),
        "policies": {
            policy_id: {
                "enabled": policies.get(policy_id).enabled,
                "checkpoint": policies.get(policy_id).checkpoint,
                "endpoint": policies.get(policy_id).endpoint,
            }
            for policy_id in policies.policy_ids()
        },
        "runtime_checked": bool(args.require_runnable),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
