"""Artifact persistence boundary."""

from abc import ABC, abstractmethod


class ArtifactStorePort(ABC):
    @abstractmethod
    def save_image(self, session_id, kind, image):
        """Persist an image and return a core.domain.ArtifactRef."""

    @abstractmethod
    def save_video(self, session_id, kind, frames, fps):
        """Persist frames and return a core.domain.ArtifactRef."""

