"""Versioned, configurable semantic vocabulary for CALVIN grounding."""

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


DEFAULT_VOCABULARY_PATH = Path(__file__).with_name("semantic_vocabulary.json")


@dataclass(frozen=True)
class CalvinSemanticVocabulary:
    name: str
    version: int
    calvin_splits: tuple
    selectors: object
    source: str

    def semantic_ids(self):
        return tuple(sorted(self.selectors))


def load_semantic_vocabulary(path=None):
    source = Path(path or DEFAULT_VOCABULARY_PATH).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            "CALVIN semantic vocabulary does not exist: {}".format(source)
        )
    try:
        data = json.loads(source.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(
            "invalid CALVIN semantic vocabulary {}: {}".format(source, exc)
        )
    errors = _validate(data)
    if errors:
        raise ValueError(
            "invalid CALVIN semantic vocabulary {}:\n{}".format(
                source, "\n".join("- {}".format(error) for error in errors)
            )
        )
    selectors = {
        semantic_id: MappingProxyType(dict(selector))
        for semantic_id, selector in data["semantics"].items()
    }
    return CalvinSemanticVocabulary(
        name=data["name"].strip(),
        version=data["version"],
        calvin_splits=tuple(data["calvin_splits"]),
        selectors=MappingProxyType(selectors),
        source=str(source),
    )


def _validate(data):
    if not isinstance(data, dict):
        return ["top-level value must be an object"]
    errors = []
    allowed = {"version", "name", "calvin_splits", "semantics"}
    unknown = sorted(set(data) - allowed)
    if unknown:
        errors.append("unknown top-level fields: {}".format(unknown))
    if data.get("version") != 1:
        errors.append("version must be 1")
    if not isinstance(data.get("name"), str) or not data.get("name", "").strip():
        errors.append("name must be a non-empty string")
    splits = data.get("calvin_splits")
    if (
        not isinstance(splits, list)
        or not splits
        or any(split not in {"A", "B", "C", "D"} for split in splits)
        or len(set(splits)) != len(splits)
    ):
        errors.append("calvin_splits must contain unique values from A, B, C, D")
    semantics = data.get("semantics")
    if not isinstance(semantics, dict) or not semantics:
        errors.append("semantics must be a non-empty object")
        return errors
    for semantic_id, selector in sorted(semantics.items()):
        path = "semantics.{}".format(semantic_id)
        if not isinstance(semantic_id, str) or not semantic_id.strip():
            errors.append("semantic IDs must be non-empty strings")
            continue
        if not isinstance(selector, dict):
            errors.append("{} must be an object".format(path))
            continue
        keys = set(selector)
        if keys == {"movable"}:
            _nonempty(selector["movable"], "{}.movable".format(path), errors)
        elif keys == {"fixed", "link"}:
            _nonempty(selector["fixed"], "{}.fixed".format(path), errors)
            _nonempty(selector["link"], "{}.link".format(path), errors)
        else:
            errors.append(
                "{} must contain either movable, or fixed and link".format(path)
            )
    return errors


def _nonempty(value, path, errors):
    if not isinstance(value, str) or not value.strip():
        errors.append("{} must be a non-empty string".format(path))
