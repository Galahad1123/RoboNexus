"""Filesystem artifact store shared by backend integrations."""

import re
from pathlib import Path
from threading import RLock

import numpy as np

from core.domain import ArtifactRef
from core.ports import ArtifactStorePort


class LocalArtifactStore(ArtifactStorePort):
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        self._lock = RLock()

    def save_image(self, session_id, kind, image):
        import imageio

        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("image must have shape HxWx3")
        if array.dtype != np.uint8:
            raise ValueError("image must have dtype uint8")
        relative = self._next_path(session_id, kind, ".png")
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(str(destination), array)
        return ArtifactRef(kind, "artifact://{}".format(relative.as_posix()), "image/png")

    def save_video(self, session_id, kind, frames, fps):
        import imageio

        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
            raise ValueError("fps must be positive")
        arrays = [np.asarray(frame) for frame in frames]
        if not arrays:
            raise ValueError("video requires at least one frame")
        relative = self._next_path(session_id, kind, ".mp4")
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(
            str(destination), arrays, fps=fps, macro_block_size=8,
            ffmpeg_log_level="error"
        )
        return ArtifactRef(kind, "artifact://{}".format(relative.as_posix()), "video/mp4")

    def resolve(self, uri):
        prefix = "artifact://"
        if not isinstance(uri, str) or not uri.startswith(prefix):
            raise ValueError("not an artifact URI: {!r}".format(uri))
        candidate = (self.root / Path(uri[len(prefix) :])).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise ValueError("artifact URI escapes store root")
        return candidate

    def _next_path(self, session_id, kind, suffix):
        with self._lock:
            self._counter += 1
            return Path(_safe(session_id)) / "{:06d}_{}{}".format(
                self._counter, _safe(kind), suffix
            )


def _safe(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("artifact path component must be a non-empty string")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())

