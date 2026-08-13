"""Stable repository paths shared by tests regardless of directory depth."""

from pathlib import Path


def _workspace_from(path):
    for parent in (path,) + tuple(path.parents):
        if (parent / "ARCHITECTURE.md").is_file():
            return parent
    raise FileNotFoundError("could not locate RoboNexus workspace")


WORKSPACE = _workspace_from(Path(__file__).resolve())
AGENT_ROOT = WORKSPACE / "RoboNexus-Agent"
