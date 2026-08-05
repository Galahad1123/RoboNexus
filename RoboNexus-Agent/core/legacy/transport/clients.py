"""HTTP clients satisfying the core model interfaces."""

from __future__ import annotations

import numpy as np

from .http import post_json
from core.legacy.perception.mask_codec import decode_mask, encode_mask, encode_rgb
from core.legacy.perception.schemas import DestinationType, GroundingResult, ObjectPrompt, ObjectRole


class QwenHttpClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.url = base_url.rstrip("/")
        self.timeout = timeout

    def locate(self, instruction: str, rgb: np.ndarray, frame_id: int) -> GroundingResult:
        data = post_json(
            f"{self.url}/locate",
            {"instruction": instruction, "frame_id": frame_id, "image_png": encode_rgb(rgb)},
            self.timeout,
        )
        objects = [
            ObjectPrompt(
                role=ObjectRole(item["role"]),
                label=item["label"],
                bbox_xyxy=tuple(item["bbox_xyxy"]),
                confidence=float(item["confidence"]),
                destination_type=DestinationType(item["destination_type"])
                if item.get("destination_type")
                else None,
                relation=item.get("relation"),
            )
            for item in data["objects"]
        ]
        return GroundingResult(data["action"], objects, int(data["frame_id"]))


class SamHttpClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.url = base_url.rstrip("/")
        self.timeout = timeout

    def segment(
        self, rgb: np.ndarray, prompts: list[ObjectPrompt], frame_id: int
    ) -> list[tuple[np.ndarray, float]]:
        data = post_json(
            f"{self.url}/segment",
            {
                "frame_id": frame_id,
                "image_png": encode_rgb(rgb),
                "prompts": [
                    {"bbox_xyxy": list(item.bbox_xyxy), "label": item.label}
                    for item in prompts
                ],
            },
            self.timeout,
        )
        return [
            (decode_mask(item["mask_png"]), float(item["confidence"]))
            for item in data["masks"]
        ]


class CutieHttpClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.url = base_url.rstrip("/")
        self.timeout = timeout

    def create_session(self, session_id: str) -> None:
        post_json(f"{self.url}/sessions/create", {"session_id": session_id}, self.timeout)

    def initialize(
        self, session_id: str, rgb: np.ndarray, masks: dict[int, np.ndarray], frame_id: int
    ) -> dict[int, tuple[np.ndarray, float]]:
        return self._frame("/sessions/initialize", session_id, rgb, frame_id, masks)

    def track(
        self, session_id: str, rgb: np.ndarray, frame_id: int
    ) -> dict[int, tuple[np.ndarray, float]]:
        return self._frame("/sessions/track", session_id, rgb, frame_id)

    def correct(
        self, session_id: str, rgb: np.ndarray, masks: dict[int, np.ndarray], frame_id: int
    ) -> dict[int, tuple[np.ndarray, float]]:
        return self._frame("/sessions/correct", session_id, rgb, frame_id, masks)

    def delete_session(self, session_id: str) -> None:
        post_json(f"{self.url}/sessions/delete", {"session_id": session_id}, self.timeout)

    def _frame(
        self,
        endpoint: str,
        session_id: str,
        rgb: np.ndarray,
        frame_id: int,
        masks: dict[int, np.ndarray] | None = None,
    ) -> dict[int, tuple[np.ndarray, float]]:
        payload = {
            "session_id": session_id,
            "frame_id": frame_id,
            "image_png": encode_rgb(rgb),
        }
        if masks is not None:
            payload["masks"] = {
                str(object_id): encode_mask(mask) for object_id, mask in masks.items()
            }
        data = post_json(f"{self.url}{endpoint}", payload, self.timeout)
        return {
            int(item["object_id"]): (
                decode_mask(item["mask_png"]),
                float(item["confidence"]),
            )
            for item in data["objects"]
        }
