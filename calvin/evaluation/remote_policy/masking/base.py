"""Mask provider contracts independent of a particular segmentation backend."""

from abc import ABC, abstractmethod


class MaskProvider(ABC):
    """Produces role-labelled masks without changing the policy task string."""

    def reset(self, episode):
        self.episode = episode

    @abstractmethod
    def masks(self, observation):
        """Return ``{camera_key: {role: bool[H,W]}}``."""


class NoMaskProvider(MaskProvider):
    def masks(self, observation):
        return {}


class TrackedMaskProvider(MaskProvider):
    """Adapter for a future SAM/Cutie-style stateful tracking backend.

    The backend owns initialization and temporal propagation. Keeping this
    adapter here makes its output interchangeable with GT masks while avoiding
    a dependency on any detector/tracker in the clean evaluation path.
    """

    def __init__(self, backend):
        self.backend = backend

    def reset(self, episode):
        super().reset(episode)
        self.backend.reset(episode)

    def masks(self, observation):
        return self.backend.masks(observation)
