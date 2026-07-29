"""Stateful inference session with explicit observation and action scheduling."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from lerobot.utils.constants import ACTION, OBS_STATE

from multitask_dit_policy.utils.utils import normalize_batch, unnormalize_batch


class InferenceSession:
    """Own one policy's episode queues and expose exactly one action per step."""

    def __init__(
        self,
        *,
        policy: Any,
        stats_tensors: dict[str, dict[str, torch.Tensor]],
        device: torch.device,
        default_replan_interval: int | None = None,
    ) -> None:
        self.policy = policy
        self.config = policy.config
        self.stats_tensors = stats_tensors
        self.device = device
        self.max_action_chunk = int(self.config.n_action_steps)
        self.default_replan_interval = default_replan_interval or self.max_action_chunk
        self.task = ""
        self.episode_id: str | None = None
        self.episode_seed: int | None = None
        self.replan_interval = self.default_replan_interval
        self.actions_since_inference = 0
        self.inference_count = 0
        self.reset()

    def reset(
        self,
        *,
        task: str = "",
        episode_id: str | None = None,
        episode_seed: int | None = None,
        replan_interval: int | None = None,
    ) -> None:
        interval = self.default_replan_interval if replan_interval is None else int(replan_interval)
        if not 1 <= interval <= self.max_action_chunk:
            raise ValueError(
                f"replan_interval must be between 1 and {self.max_action_chunk}, got {interval}"
            )
        self.policy.reset()
        self.task = str(task)
        self.episode_id = episode_id
        self.episode_seed = None if episode_seed is None else int(episode_seed)
        if self.episode_seed is not None:
            if self.episode_seed < 0:
                raise ValueError(f"episode_seed must be non-negative, got {self.episode_seed}")
            torch.manual_seed(self.episode_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.episode_seed)
        self.replan_interval = interval
        self.actions_since_inference = 0
        self.inference_count = 0

    @torch.no_grad()
    def step(self, observation: dict[str, Any], step: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        del step
        must_replan = (
            len(self.policy._queues[ACTION]) == 0
            or self.actions_since_inference >= self.replan_interval
        )
        if must_replan:
            # Keep n_obs_steps history, but discard the unused tail of the
            # previous action chunk before select_action generates a new one.
            self.policy._queues[ACTION].clear()
            self.actions_since_inference = 0

        batch = self._observation_to_batch(observation)
        normalized = normalize_batch(
            batch,
            self.config.input_features,
            self.config.output_features,
            self.config.normalization_mapping,
            self.stats_tensors,
        )
        action = self.policy.select_action(normalized)
        if must_replan:
            self.inference_count += 1
        self.actions_since_inference += 1

        if action.ndim == 2:
            action = action.squeeze(0)
        unnormalized = unnormalize_batch(
            {ACTION: action.reshape(1, 1, -1)},
            self.config.output_features,
            self.config.normalization_mapping,
            self.stats_tensors,
            feature_keys=[ACTION],
        )[ACTION].reshape(-1)
        action_np = unnormalized.detach().cpu().numpy().astype(np.float32)
        metadata = {
            "replanned": must_replan,
            "inference_count": self.inference_count,
            "action_index_in_plan": self.actions_since_inference - 1,
            "replan_interval": self.replan_interval,
            "episode_seed": self.episode_seed,
        }
        return action_np, metadata

    def _observation_to_batch(self, observation: dict[str, Any]) -> dict[str, Any]:
        state_dim = int(self.config.robot_state_feature.shape[0])
        state = observation.get(OBS_STATE, observation.get("state"))
        if state is None:
            raise KeyError(f"Request is missing {OBS_STATE}.")
        state_tensor = torch.as_tensor(
            np.asarray(state).copy(),
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)
        if state_tensor.shape[-1] != state_dim:
            raise ValueError(f"Expected state dim {state_dim}, got {state_tensor.shape[-1]}.")

        batch: dict[str, Any] = {OBS_STATE: state_tensor, "task": [self.task]}
        image_keys = list(self.config.image_features)
        missing = [key for key in image_keys if key not in observation]
        if missing:
            raise KeyError(f"Request missing image key(s): {missing}.")
        for key in image_keys:
            batch[key] = self._image_to_tensor(observation[key], key)
        return batch

    def _image_to_tensor(self, image: Any, key: str) -> torch.Tensor:
        array = np.asarray(image).copy()
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3:
            raise ValueError(f"Expected {key} as HWC or CHW, got {array.shape}.")

        channels, height, width = self.config.input_features[key].shape
        if array.shape[-1] == channels:
            array = np.transpose(array, (2, 0, 1))
        elif array.shape[0] != channels:
            raise ValueError(f"Expected {channels} channels for {key}, got {array.shape}.")
        if array.shape[-2:] != (height, width):
            raise ValueError(
                f"Expected {key} resolution {(height, width)}, got {array.shape[-2:]}. "
                "Environment clients must match the checkpoint schema."
            )

        tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device).unsqueeze(0)
        stats = self.stats_tensors.get(key, {})
        max_stat = stats.get("max")
        if max_stat is not None and torch.max(max_stat).item() <= 1.5 and tensor.max().item() > 1.5:
            tensor = tensor / 255.0
        return tensor
