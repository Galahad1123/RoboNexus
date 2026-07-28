from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MockSession:
    masks: dict[int, np.ndarray]


class MockCutieBackend:
    def __init__(self) -> None:
        self.sessions: dict[str, MockSession] = {}

    def create_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            raise ValueError(f"session already exists: {session_id}")
        self.sessions[session_id] = MockSession({})

    def initialize(
        self,
        session_id: str,
        rgb: np.ndarray,
        masks: dict[int, np.ndarray],
        frame_id: int | None = None,
    ) -> dict[int, tuple[np.ndarray, float]]:
        self.sessions[session_id].masks = {key: value.copy() for key, value in masks.items()}
        return {key: (value.copy(), 1.0) for key, value in masks.items()}

    def track(
        self, session_id: str, rgb: np.ndarray, frame_id: int | None = None
    ) -> dict[int, tuple[np.ndarray, float]]:
        return {
            key: (value.copy(), 0.95)
            for key, value in self.sessions[session_id].masks.items()
        }

    def correct(
        self,
        session_id: str,
        rgb: np.ndarray,
        masks: dict[int, np.ndarray],
        frame_id: int | None = None,
    ) -> dict[int, tuple[np.ndarray, float]]:
        self.sessions[session_id].masks.update(
            {key: value.copy() for key, value in masks.items()}
        )
        return {
            key: (value.copy(), 1.0)
            for key, value in self.sessions[session_id].masks.items()
        }

    def delete_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)


@dataclass
class CutieSession:
    processor: object
    object_ids: list[int]


class CutieBackend:
    def __init__(self, max_internal_size: int = 480, device: str = "cuda") -> None:
        import torch
        from cutie.utils.get_default_model import get_default_model

        self.torch = torch
        self.device = device
        self.model = get_default_model().to(device).eval()
        self.max_internal_size = max_internal_size
        self.sessions: dict[str, CutieSession] = {}

    def create_session(self, session_id: str) -> None:
        from cutie.inference.inference_core import InferenceCore

        if session_id in self.sessions:
            raise ValueError(f"session already exists: {session_id}")
        processor = InferenceCore(self.model, cfg=self.model.cfg)
        processor.max_internal_size = self.max_internal_size
        self.sessions[session_id] = CutieSession(processor, [])

    def initialize(
        self,
        session_id: str,
        rgb: np.ndarray,
        masks: dict[int, np.ndarray],
        frame_id: int | None = None,
    ) -> dict[int, tuple[np.ndarray, float]]:
        session = self.sessions[session_id]
        session.object_ids = sorted(masks)
        index_mask = self._index_mask(masks)
        probability = self._step(session, rgb, index_mask, session.object_ids)
        return self._results(session, probability)

    def track(
        self, session_id: str, rgb: np.ndarray, frame_id: int | None = None
    ) -> dict[int, tuple[np.ndarray, float]]:
        session = self.sessions[session_id]
        return self._results(session, self._step(session, rgb))

    def correct(
        self,
        session_id: str,
        rgb: np.ndarray,
        masks: dict[int, np.ndarray],
        frame_id: int | None = None,
    ) -> dict[int, tuple[np.ndarray, float]]:
        session = self.sessions[session_id]
        for object_id in masks:
            if object_id not in session.object_ids:
                session.object_ids.append(object_id)
        session.object_ids.sort()
        probability = self._step(
            session, rgb, self._index_mask(masks), sorted(masks), force_permanent=True
        )
        return self._results(session, probability)

    def delete_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()

    def _step(
        self,
        session: CutieSession,
        rgb: np.ndarray,
        mask: np.ndarray | None = None,
        objects: list[int] | None = None,
        force_permanent: bool = False,
    ):
        image = self.torch.from_numpy(rgb).permute(2, 0, 1).float().to(self.device) / 255.0
        tensor_mask = (
            self.torch.from_numpy(mask).long().to(self.device) if mask is not None else None
        )
        with self.torch.inference_mode(), self.torch.autocast(
            device_type=self.device.split(":")[0], enabled=self.device.startswith("cuda")
        ):
            return session.processor.step(
                image,
                tensor_mask,
                objects=objects,
                force_permanent=force_permanent,
            )

    def _results(self, session: CutieSession, probability) -> dict[int, tuple[np.ndarray, float]]:
        label_mask = session.processor.output_prob_to_mask(probability).cpu().numpy()
        results = {}
        for object_id in session.object_ids:
            binary = (label_mask == object_id).astype(np.uint8)
            temp_id = session.processor.object_manager.find_tmp_by_id(object_id)
            region = self.torch.from_numpy(binary.astype(bool)).to(probability.device)
            confidence = (
                float(probability[temp_id][region].mean().detach().cpu())
                if region.any()
                else 0.0
            )
            results[object_id] = (binary, confidence)
        return results

    @staticmethod
    def _index_mask(masks: dict[int, np.ndarray]) -> np.ndarray:
        shape = next(iter(masks.values())).shape
        result = np.zeros(shape, dtype=np.int64)
        for object_id in sorted(masks):
            result[masks[object_id] > 0] = object_id
        return result
