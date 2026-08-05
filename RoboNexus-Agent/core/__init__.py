"""Model-independent RoboNexus core.

The package root intentionally uses lazy compatibility exports. This lets the
Python-3.8 CALVIN runtime import isolated control/config subpackages without
eagerly importing the newer perception schemas.
"""

_PERCEPTION_EXPORTS = {
    "FramePacket": ("core.legacy.perception.schemas", "FramePacket"),
    "GroundingResult": ("core.legacy.perception.schemas", "GroundingResult"),
    "ObjectPrompt": ("core.legacy.perception.schemas", "ObjectPrompt"),
    "ObjectRole": ("core.legacy.perception.schemas", "ObjectRole"),
    "PerceptionPipeline": ("core.legacy.perception.orchestrator", "PerceptionPipeline"),
    "PipelineResult": ("core.legacy.perception.schemas", "PipelineResult"),
    "PipelineState": ("core.legacy.perception.schemas", "PipelineState"),
    "TrackedObject": ("core.legacy.perception.schemas", "TrackedObject"),
}

__all__ = sorted(_PERCEPTION_EXPORTS)


def __getattr__(name):
    if name not in _PERCEPTION_EXPORTS:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    import importlib

    module_name, attribute = _PERCEPTION_EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value
