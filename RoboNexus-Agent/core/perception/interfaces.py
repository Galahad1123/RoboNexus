"""Structural interfaces implemented by local and HTTP model clients."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..common.schemas import GroundingResult, ObjectPrompt


class Grounder(Protocol):
    def locate(self, instruction: str, rgb: np.ndarray, frame_id: int) -> GroundingResult: ...


class Segmenter(Protocol):
    def segment(
        self, rgb: np.ndarray, prompts: list[ObjectPrompt], frame_id: int
    ) -> list[tuple[np.ndarray, float]]: ...


class Tracker(Protocol):
    def create_session(self, session_id: str) -> None: ...
    def initialize(
        self, session_id: str, rgb: np.ndarray, masks: dict[int, np.ndarray], frame_id: int
    ) -> dict[int, tuple[np.ndarray, float]]: ...
    def track(
        self, session_id: str, rgb: np.ndarray, frame_id: int
    ) -> dict[int, tuple[np.ndarray, float]]: ...
    def correct(
        self, session_id: str, rgb: np.ndarray, masks: dict[int, np.ndarray], frame_id: int
    ) -> dict[int, tuple[np.ndarray, float]]: ...
    def delete_session(self, session_id: str) -> None: ...
