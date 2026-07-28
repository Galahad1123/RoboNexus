from __future__ import annotations

from typing import Any

import numpy as np


class MockSamBackend:
    def segment(self, rgb: np.ndarray, prompts: list[dict[str, Any]]) -> list[tuple[np.ndarray, float]]:
        height, width = rgb.shape[:2]
        results = []
        for prompt in prompts:
            x1, y1, x2, y2 = map(int, prompt["bbox_xyxy"])
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = 1
            results.append((mask, 0.9))
        return results


class SamBackend:
    def __init__(self, checkpoint: str, model_type: str = "vit_b", device: str = "cuda") -> None:
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        self.torch = torch
        model = sam_model_registry[model_type](checkpoint=checkpoint).to(device).eval()
        self.predictor = SamPredictor(model)

    def segment(self, rgb: np.ndarray, prompts: list[dict[str, Any]]) -> list[tuple[np.ndarray, float]]:
        self.predictor.set_image(rgb)
        results = []
        for prompt in prompts:
            box = np.asarray(prompt["bbox_xyxy"], dtype=np.float32)
            with self.torch.inference_mode():
                masks, scores, _ = self.predictor.predict(
                    box=box, multimask_output=True, return_logits=False
                )
            best = int(np.argmax(scores))
            results.append((masks[best].astype(np.uint8), float(scores[best])))
        return results
