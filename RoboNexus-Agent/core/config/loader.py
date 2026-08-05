"""Strict, Python-3.8-compatible YAML loading helpers."""

from pathlib import Path


class ConfigValidationError(ValueError):
    """A deterministic collection of user-fixable configuration errors."""

    def __init__(self, source, errors):
        self.source = str(source)
        self.errors = tuple(str(error) for error in errors)
        message = "Invalid configuration {}:\n{}".format(
            self.source,
            "\n".join("- {}".format(error) for error in self.errors),
        )
        super().__init__(message)


def load_yaml_mapping(path):
    """Load a YAML object while rejecting duplicate mapping keys."""
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Configuration loading requires PyYAML in the active environment"
        ) from exc

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("Configuration file does not exist: {}".format(source))

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ConfigValidationError(
                    source,
                    ["duplicate key {!r} at line {}".format(key, key_node.start_mark.line + 1)],
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    try:
        data = yaml.load(source.read_text(), Loader=UniqueKeyLoader)
    except ConfigValidationError:
        raise
    except yaml.YAMLError as exc:
        raise ConfigValidationError(source, ["invalid YAML: {}".format(exc)]) from exc
    if not isinstance(data, dict):
        raise ConfigValidationError(source, ["top-level value must be a mapping"])
    return source, data
