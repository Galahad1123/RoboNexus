"""Qwen3-VL and deterministic mock grounding backends."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
from PIL import Image


SYSTEM_PROMPT = """You ground robot manipulation instructions in one RGB image.
Return JSON only with keys action and objects. Each object has role
(source or destination), label, bbox_xyxy in ORIGINAL image pixel coordinates,
confidence in [0,1], and optional destination_type (object, region, relation)
and relation. Use half-open boxes. Include the manipulated source and, when
present, its placement/reference destination. Never invent an invisible box."""


class MockQwenBackend:
    def locate(self, instruction: str, rgb: np.ndarray) -> dict[str, Any]:
        height, width = rgb.shape[:2]
        lower = instruction.lower()
        destination_words = ("place", "put", "stack", "into", "in drawer", "放", "堆")
        objects = [
            {
                "role": "source",
                "label": instruction,
                "bbox_xyxy": [width // 8, height // 4, width // 2, 3 * height // 4],
                "confidence": 0.8,
            }
        ]
        if any(word in lower for word in destination_words):
            objects.append(
                {
                    "role": "destination",
                    "label": "destination",
                    "bbox_xyxy": [width // 2, height // 4, 7 * width // 8, 3 * height // 4],
                    "confidence": 0.75,
                    "destination_type": "region",
                }
            )
        return {"action": lower.split(" ", 1)[0] or "manipulate", "objects": objects}


class QwenBackend:
    def __init__(self, model_path: str, max_new_tokens: int = 512) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto"
        ).eval()
        self.max_new_tokens = max_new_tokens

    def locate(self, instruction: str, rgb: np.ndarray) -> dict[str, Any]:
        height, width = rgb.shape[:2]
        user_text = (
            f"Instruction: {instruction}\nImage size: {width}x{height}. "
            "Ground every relevant role."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.fromarray(rgb)},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Qwen did not return a JSON object: {text[:300]}")
        result = json.loads(match.group(0))
        self._validate(result, width, height)
        return result

    @staticmethod
    def _validate(result: dict[str, Any], width: int, height: int) -> None:
        if not isinstance(result.get("action"), str) or not result.get("objects"):
            raise ValueError("grounding JSON requires action and non-empty objects")
        for item in result["objects"]:
            if item.get("role") not in {"source", "destination", "tool", "obstacle"}:
                raise ValueError(f"invalid role: {item.get('role')}")
            x1, y1, x2, y2 = map(int, item["bbox_xyxy"])
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"out-of-range bbox: {item['bbox_xyxy']}")
            item["bbox_xyxy"] = [x1, y1, x2, y2]
            item["confidence"] = float(item["confidence"])
