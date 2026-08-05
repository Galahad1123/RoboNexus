"""Robot/simulator observation boundary."""

from abc import ABC, abstractmethod


class EnvironmentPortError(RuntimeError):
    """The environment cannot provide a valid current scene."""


class EnvironmentPort(ABC):
    @abstractmethod
    def capture_scene(self):
        """Return current static RGB, wrist RGB and optional state summaries."""

    @property
    @abstractmethod
    def environment_steps(self):
        """Return environment steps executed in the current Agent session."""
