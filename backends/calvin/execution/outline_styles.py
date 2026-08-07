"""Load policy-specific outline styles matching specialist training data."""

import json
from pathlib import Path

from calvin.evaluation.remote_policy.masking.overlay import MaskOverlay


DEFAULT_OUTLINE_STYLES_PATH = Path(__file__).with_name("outline_styles.json")


def load_outline_styles(path=None):
    source = Path(path or DEFAULT_OUTLINE_STYLES_PATH).expanduser().resolve()
    data = json.loads(source.read_text())
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("outline style config version must be 1")
    policies = data.get("policies")
    if not isinstance(policies, dict) or not policies:
        raise ValueError("outline style config requires policies")
    result = {}
    for policy_id, style in policies.items():
        if not isinstance(policy_id, str) or not policy_id or not isinstance(style, dict):
            raise ValueError("invalid outline style for {!r}".format(policy_id))
        try:
            result[policy_id] = MaskOverlay(**style)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "invalid outline style for {!r}: {}".format(policy_id, exc)
            )
    return result
