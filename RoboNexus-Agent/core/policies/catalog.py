"""Strict policy placeholder configuration loading and validation."""

from pathlib import Path
from types import MappingProxyType

from core.config import ConfigValidationError, load_yaml_mapping

from .schemas import PolicySpec


class PolicyCatalog:
    def __init__(self, name, version, checkpoint_base, policies, source):
        self.name = name
        self.version = version
        self.checkpoint_base = checkpoint_base
        self.source = source
        self._policies = MappingProxyType(dict(policies))

    def get(self, policy_id):
        try:
            return self._policies[policy_id]
        except KeyError as exc:
            raise KeyError("Unknown policy_id {!r}".format(policy_id)) from exc

    def policy_ids(self):
        return tuple(sorted(self._policies))

    def resolve_checkpoint(self, policy_id, workspace):
        policy = self.get(policy_id)
        if not policy.checkpoint:
            return None
        checkpoint = Path(policy.checkpoint).expanduser()
        if checkpoint.is_absolute():
            return checkpoint.resolve()
        return (Path(workspace) / self.checkpoint_base / checkpoint).resolve()

    def validate_runtime(self, workspace, policy_ids=None):
        errors = []
        selected = tuple(policy_ids or self.policy_ids())
        for policy_id in selected:
            policy = self.get(policy_id)
            prefix = "policies.{}".format(policy_id)
            if not policy.enabled:
                errors.append("{} is disabled".format(prefix))
                continue
            checkpoint = self.resolve_checkpoint(policy_id, workspace)
            if checkpoint is None:
                errors.append("{}.checkpoint is not configured".format(prefix))
                continue
            if not checkpoint.is_dir():
                errors.append("{}.checkpoint does not exist: {}".format(prefix, checkpoint))
                continue
            for filename in ("config.json", "dataset_stats.json", "model.safetensors"):
                if not (checkpoint / filename).is_file():
                    errors.append("{}.checkpoint is missing {}".format(prefix, filename))
        if errors:
            raise ConfigValidationError(self.source, errors)


def load_policy_catalog(path, required_policy_ids=None):
    source, data = load_yaml_mapping(path)
    errors = []
    unknown_top_level = sorted(
        set(data) - {"version", "name", "checkpoint_base", "policies"}
    )
    if unknown_top_level:
        errors.append("unknown top-level fields: {}".format(unknown_top_level))
    if data.get("version") != 1:
        errors.append("version must be 1")
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("name must be a non-empty string")
    checkpoint_base = data.get("checkpoint_base")
    if not isinstance(checkpoint_base, str) or not checkpoint_base.strip():
        errors.append("checkpoint_base must be a non-empty string")

    raw_policies = data.get("policies")
    if not isinstance(raw_policies, dict):
        errors.append("policies must be a mapping")
        raw_policies = {}
    actual_ids = set(raw_policies)
    if required_policy_ids is not None:
        required = set(required_policy_ids)
        missing = sorted(required - actual_ids)
        extra = sorted(actual_ids - required)
        if missing:
            errors.append("policies is missing required IDs: {}".format(missing))
        if extra:
            errors.append("policies has unknown IDs: {}".format(extra))

    policies = {}
    endpoints = {}
    allowed_fields = {"enabled", "checkpoint", "host", "port"}
    for policy_id in sorted(raw_policies):
        raw = raw_policies[policy_id]
        prefix = "policies.{}".format(policy_id)
        if not isinstance(raw, dict):
            errors.append("{} must be a mapping".format(prefix))
            continue
        unknown = sorted(set(raw) - allowed_fields)
        if unknown:
            errors.append("{} has unknown fields: {}".format(prefix, unknown))
        enabled = raw.get("enabled")
        checkpoint = raw.get("checkpoint")
        host = raw.get("host")
        port = raw.get("port")
        if not isinstance(enabled, bool):
            errors.append("{}.enabled must be boolean".format(prefix))
        if checkpoint is not None and (
            not isinstance(checkpoint, str) or not checkpoint.strip()
        ):
            errors.append("{}.checkpoint must be null or a non-empty string".format(prefix))
        if enabled is True and checkpoint is None:
            errors.append("{}.checkpoint is required when enabled=true".format(prefix))
        if not isinstance(host, str) or not host.strip():
            errors.append("{}.host must be a non-empty string".format(prefix))
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            errors.append("{}.port must be an integer in [1, 65535]".format(prefix))
        if isinstance(host, str) and host.strip() and isinstance(port, int) and not isinstance(port, bool):
            endpoint = (host, port)
            if endpoint in endpoints:
                errors.append(
                    "{} duplicates endpoint used by policies.{}".format(
                        prefix, endpoints[endpoint]
                    )
                )
            else:
                endpoints[endpoint] = policy_id
        if (
            isinstance(enabled, bool)
            and (checkpoint is None or isinstance(checkpoint, str))
            and isinstance(host, str)
            and host.strip()
            and isinstance(port, int)
            and not isinstance(port, bool)
            and 1 <= port <= 65535
        ):
            policies[policy_id] = PolicySpec(
                policy_id=policy_id,
                enabled=enabled,
                checkpoint=checkpoint,
                host=host,
                port=port,
            )

    if errors:
        raise ConfigValidationError(source, errors)
    return PolicyCatalog(name, 1, checkpoint_base, policies, source)
