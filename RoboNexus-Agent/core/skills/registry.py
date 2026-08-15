"""Strict loader and immutable lookup registry for atomic skills."""

from types import MappingProxyType

from core.config import ConfigValidationError, load_yaml_mapping

from .calvin_d34 import CALVIN_D34_ROUTING, CALVIN_D34_SKILL_IDS
from .schemas import PlannerGuidance, RecoveryRule, SkillDefaults, SkillSpec


POSITIVE_INTEGER_FIELDS = ("max_steps", "max_attempts", "replan_interval")
SKILL_CATEGORIES = {
    "lift", "place", "push", "rotate", "articulation", "switch", "stack"
}


class SkillRegistry:
    def __init__(self, name, version, defaults, skills, source):
        self.name = name
        self.version = version
        self.defaults = defaults
        self.source = source
        self._skills = MappingProxyType(dict(skills))

    def get(self, skill_id):
        try:
            return self._skills[skill_id]
        except KeyError as exc:
            raise KeyError("Unknown skill_id {!r}".format(skill_id)) from exc

    def __contains__(self, skill_id):
        return skill_id in self._skills

    def __len__(self):
        return len(self._skills)

    def skill_ids(self):
        return tuple(sorted(self._skills))

    def policy_groups(self):
        return tuple(sorted({skill.policy_group for skill in self._skills.values()}))

    def by_policy_group(self, policy_group):
        return tuple(
            self._skills[skill_id]
            for skill_id in sorted(self._skills)
            if self._skills[skill_id].policy_group == policy_group
        )

    def categories(self):
        return tuple(sorted({skill.category for skill in self._skills.values()}))

    def by_category(self, category):
        return tuple(
            self._skills[skill_id]
            for skill_id in sorted(self._skills)
            if self._skills[skill_id].category == category
        )


def _positive_integer(value, path, errors):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        errors.append("{} must be a positive integer".format(path))
        return None
    return value


def _success_guard(value, path, errors):
    if value is None:
        return None
    if not isinstance(value, dict):
        errors.append("{} must be a mapping".format(path))
        return None
    allowed = {"gripper", "min_width_m", "consecutive_steps"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        errors.append("{} has unknown fields: {}".format(path, unknown))
    if value.get("gripper") not in {"open", "closed"}:
        errors.append("{}.gripper must be 'open' or 'closed'".format(path))
    width = value.get("min_width_m")
    if isinstance(width, bool) or not isinstance(width, (int, float)) or width < 0:
        errors.append("{}.min_width_m must be a non-negative number".format(path))
    _positive_integer(value.get("consecutive_steps"), "{}.consecutive_steps".format(path), errors)
    return dict(value)


def _planner_guidance(value, path, errors):
    if value is None:
        return None
    if not isinstance(value, dict):
        errors.append("{} must be a mapping".format(path))
        return None
    allowed = {"planning", "grounding", "recovery"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        errors.append("{} has unknown fields: {}".format(path, unknown))
    parsed = {}
    for field in sorted(allowed):
        notes = value.get(field, [])
        if (
            not isinstance(notes, list)
            or any(not isinstance(note, str) or not note.strip() for note in notes)
        ):
            errors.append("{}.{} must be a list of non-empty strings".format(path, field))
            parsed[field] = ()
        else:
            parsed[field] = tuple(note.strip() for note in notes)
    if not any(parsed.values()):
        errors.append("{} must contain at least one guidance note".format(path))
    return PlannerGuidance(**parsed)


def _string_list(value, path, errors):
    if value is None:
        return ()
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        errors.append("{} must be a list of non-empty strings".format(path))
        return ()
    return tuple(item.strip() for item in value)


def _recovery_rules(value, path, errors):
    if value is None:
        return ()
    if not isinstance(value, list):
        errors.append("{} must be a list".format(path))
        return ()
    result = []
    allowed = {
        "condition", "strategy", "candidate_skills", "candidate_categories",
        "guidance", "return_to_skill",
    }
    for index, raw in enumerate(value):
        item_path = "{}[{}]".format(path, index)
        if not isinstance(raw, dict):
            errors.append("{} must be a mapping".format(item_path))
            continue
        unknown = sorted(set(raw) - allowed)
        if unknown:
            errors.append("{} has unknown fields: {}".format(item_path, unknown))
        condition = raw.get("condition")
        strategy = raw.get("strategy")
        if not isinstance(condition, str) or not condition.strip():
            errors.append("{}.condition must be a non-empty string".format(item_path))
            condition = "invalid"
        if not isinstance(strategy, str) or not strategy.strip():
            errors.append("{}.strategy must be a non-empty string".format(item_path))
            strategy = "invalid"
        skills = _string_list(raw.get("candidate_skills"), "{}.candidate_skills".format(item_path), errors)
        categories = _string_list(raw.get("candidate_categories"), "{}.candidate_categories".format(item_path), errors)
        for category in categories:
            if category not in SKILL_CATEGORIES:
                errors.append("{}.candidate_categories contains unknown category {!r}".format(item_path, category))
        guidance = _string_list(raw.get("guidance"), "{}.guidance".format(item_path), errors)
        return_to = raw.get("return_to_skill")
        if return_to is not None and (not isinstance(return_to, str) or not return_to.strip()):
            errors.append("{}.return_to_skill must be a non-empty string".format(item_path))
            return_to = None
        result.append(RecoveryRule(condition.strip(), strategy.strip(), skills, categories, guidance, return_to))
    return tuple(result)


def load_skill_registry(
    path,
    expected_skill_ids=CALVIN_D34_SKILL_IDS,
    expected_routing=CALVIN_D34_ROUTING,
):
    source, data = load_yaml_mapping(path)
    errors = []
    unknown_top_level = sorted(set(data) - {"version", "name", "defaults", "skills"})
    if unknown_top_level:
        errors.append("unknown top-level fields: {}".format(unknown_top_level))
    if data.get("version") != 1:
        errors.append("version must be 1")
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("name must be a non-empty string")

    raw_defaults = data.get("defaults")
    if not isinstance(raw_defaults, dict):
        errors.append("defaults must be a mapping")
        raw_defaults = {}
    unknown_defaults = sorted(set(raw_defaults) - set(POSITIVE_INTEGER_FIELDS))
    if unknown_defaults:
        errors.append("defaults has unknown fields: {}".format(unknown_defaults))
    default_values = {
        field: _positive_integer(raw_defaults.get(field), "defaults.{}".format(field), errors)
        for field in POSITIVE_INTEGER_FIELDS
    }

    raw_skills = data.get("skills")
    if not isinstance(raw_skills, dict):
        errors.append("skills must be a mapping")
        raw_skills = {}
    actual_ids = set(raw_skills)
    if expected_skill_ids is not None:
        expected_ids = set(expected_skill_ids)
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        if missing:
            errors.append("skills is missing canonical IDs: {}".format(missing))
        if extra:
            errors.append("skills has unknown IDs: {}".format(extra))

    skills = {}
    allowed_fields = {
        "instruction",
        "oracle_task",
        "policy_group",
        "category",
        "grounding_roles",
        "max_steps",
        "max_attempts",
        "replan_interval",
        "success_guard",
        "planner_guidance",
        "recovery",
    }
    for skill_id in sorted(raw_skills):
        raw = raw_skills[skill_id]
        prefix = "skills.{}".format(skill_id)
        if not isinstance(skill_id, str) or not skill_id:
            errors.append("skill IDs must be non-empty strings")
            continue
        if not isinstance(raw, dict):
            errors.append("{} must be a mapping".format(prefix))
            continue
        unknown = sorted(set(raw) - allowed_fields)
        if unknown:
            errors.append("{} has unknown fields: {}".format(prefix, unknown))
        instruction = raw.get("instruction")
        oracle_task = raw.get("oracle_task")
        policy_group = raw.get("policy_group")
        category = raw.get("category")
        grounding_roles = raw.get("grounding_roles")
        if not isinstance(instruction, str) or not instruction.strip():
            errors.append("{}.instruction must be a non-empty string".format(prefix))
        if not isinstance(oracle_task, str) or not oracle_task.strip():
            errors.append("{}.oracle_task must be a non-empty string".format(prefix))
        elif oracle_task != skill_id:
            errors.append("{}.oracle_task must equal its canonical skill ID".format(prefix))
        if not isinstance(policy_group, str) or not policy_group.strip():
            errors.append("{}.policy_group must be a non-empty string".format(prefix))
        elif expected_routing is not None:
            expected_group = next(
                (
                    group
                    for group, group_skill_ids in expected_routing.items()
                    if skill_id in group_skill_ids
                ),
                None,
            )
            if expected_group is not None and policy_group != expected_group:
                errors.append(
                    "{}.policy_group must be {!r}, got {!r}".format(
                        prefix, expected_group, policy_group
                    )
                )
        if category not in SKILL_CATEGORIES:
            errors.append("{}.category must be one of {}".format(prefix, sorted(SKILL_CATEGORIES)))
        allowed_roles = {"source", "destination", "target"}
        if (
            not isinstance(grounding_roles, list)
            or not 1 <= len(grounding_roles) <= 2
            or any(role not in allowed_roles for role in grounding_roles)
            or len(set(grounding_roles)) != len(grounding_roles)
            or ("target" in grounding_roles and len(grounding_roles) != 1)
        ):
            errors.append(
                "{}.grounding_roles must contain one or two unique source/destination "
                "roles, or target alone".format(prefix)
            )
        values = {}
        for field in POSITIVE_INTEGER_FIELDS:
            value = raw.get(field, default_values.get(field))
            values[field] = _positive_integer(value, "{}.{}".format(prefix, field), errors)
        guard = _success_guard(raw.get("success_guard"), "{}.success_guard".format(prefix), errors)
        guidance = _planner_guidance(
            raw.get("planner_guidance"), "{}.planner_guidance".format(prefix), errors
        )
        recovery = _recovery_rules(raw.get("recovery"), "{}.recovery".format(prefix), errors)
        if all(
            [
                isinstance(instruction, str) and bool(instruction.strip()),
                isinstance(oracle_task, str) and bool(oracle_task.strip()),
                isinstance(policy_group, str) and bool(policy_group.strip()),
            ]
        ) and all(value is not None for value in values.values()):
            skills[skill_id] = SkillSpec(
                skill_id=skill_id,
                instruction=instruction,
                oracle_task=oracle_task,
                policy_group=policy_group,
                category=category,
                grounding_roles=tuple(grounding_roles or ()),
                max_steps=values["max_steps"],
                max_attempts=values["max_attempts"],
                replan_interval=values["replan_interval"],
                success_guard=guard,
                planner_guidance=guidance,
                recovery=recovery,
            )

    for skill in skills.values():
        for index, rule in enumerate(skill.recovery):
            for candidate in rule.candidate_skills:
                if candidate not in skills:
                    errors.append(
                        "skills.{}.recovery[{}].candidate_skills references unknown skill {!r}".format(
                            skill.skill_id, index, candidate
                        )
                    )
            if rule.return_to_skill is not None and rule.return_to_skill not in skills:
                errors.append(
                    "skills.{}.recovery[{}].return_to_skill references unknown skill {!r}".format(
                        skill.skill_id, index, rule.return_to_skill
                    )
                )
    if errors:
        raise ConfigValidationError(source, errors)
    defaults = SkillDefaults(**default_values)
    return SkillRegistry(name, 1, defaults, skills, source)
