#!/usr/bin/env python

# Copyright 2025 Bryson Jones and the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Training script for Multi-Task Diffusion Transformer (DiT) policy.

Credit is given to the https://github.com/huggingface/lerobot project
for which this training script is adapted from.
"""

import logging
import os
import random
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import wandb
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from multitask_dit_policy.model.model import MultiTaskDiTPolicy
from multitask_dit_policy.utils.configuration import MultiTaskDiTConfig
from multitask_dit_policy.utils.utils import move_to_device, normalize_batch, save_policy

# Suppress Pydantic warnings from draccus ChoiceRegistry union types
# This is an interaction with draccus that we can't control
warnings.filterwarnings("ignore", message=".*Field.*attribute.*repr.*")
warnings.filterwarnings("ignore", message=".*Field.*attribute.*frozen.*")
warnings.filterwarnings("ignore", module="pydantic._internal._generate_schema")

import draccus  # noqa: E402


@dataclass
class TrainConfig:
    # Dataset parameters
    dataset_path: str
    val_dataset_path: str | None = None
    probe_dataset_path: str | None = None
    video_backend: str = "torchcodec"

    # Training parameters
    batch_size: int = 16
    num_workers: int = 2
    train_steps: int = 10_000
    save_freq: int = 500
    log_freq: int = 25
    val_ratio: float = 0.05
    val_freq: int = 500
    val_batches: int = 10
    probe_freq: int = 500
    probe_batches: int = 10
    output_dir: str = "outputs/train_multi_task_dit"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = False
    use_wandb: bool = False
    seed: int = 17
    task_sampling: str = "proportional"

    # Checkpoint loading
    checkpoint_path: str | None = None

    # Policy parameters
    policy: MultiTaskDiTConfig = field(default_factory=MultiTaskDiTConfig)


def _cycle(iterable):
    """Yield from a re-iterable forever without caching its batches."""
    while True:
        yield from iterable


def _to_episode_subset(values, episode_indices: list[int]):
    """Select episode rows while preserving pandas/Polars/list compatibility."""
    return values.iloc[episode_indices] if hasattr(values, "iloc") else values[episode_indices]


def _split_episode_indices(num_episodes: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Create a deterministic, leakage-free split at episode granularity."""
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if val_ratio == 0 or num_episodes < 2:
        return list(range(num_episodes)), []

    num_val = max(1, int(num_episodes * val_ratio))
    num_val = min(num_val, num_episodes - 1)

    rng = random.Random(seed)
    indices = list(range(num_episodes))
    rng.shuffle(indices)

    val_indices = sorted(indices[:num_val])
    train_indices = sorted(indices[num_val:])
    return train_indices, val_indices


def _validate_validation_config(val_dataset_path: str | None, val_ratio: float) -> None:
    """Reject ambiguous internal/external validation configurations."""
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if val_dataset_path is not None and val_ratio != 0:
        raise ValueError("val_ratio must be 0 when val_dataset_path is provided")


def _validate_dataset_compatibility(
    train_metadata: LeRobotDatasetMetadata,
    val_metadata: LeRobotDatasetMetadata,
    feature_keys: set[str],
) -> None:
    """Ensure an external validation dataset can be sampled like the training dataset."""
    if train_metadata.fps != val_metadata.fps:
        raise ValueError(
            f"Training and validation dataset FPS differ: {train_metadata.fps} != {val_metadata.fps}"
        )

    errors = []
    for key in sorted(feature_keys):
        train_feature = train_metadata.features.get(key)
        val_feature = val_metadata.features.get(key)
        if train_feature is None or val_feature is None:
            errors.append(f"{key}: missing from {'training' if train_feature is None else 'validation'} dataset")
            continue
        for field in ("dtype", "shape"):
            if train_feature[field] != val_feature[field]:
                errors.append(
                    f"{key}.{field}: training={train_feature[field]!r}, validation={val_feature[field]!r}"
                )
    if errors:
        raise ValueError("Training and validation dataset features are incompatible:\n" + "\n".join(errors))


def _build_task_balanced_sampler(
    episodes,
    episode_indices: list[int],
    drop_n_last_frames: int,
    seed: int,
) -> object:
    """Sample valid frames so each episode-level task has equal probability.

    CALVIN-D-10 episodes contain one task. A tuple key also makes the behavior
    deterministic and well-defined for datasets that attach multiple tasks to
    an episode.
    """
    rows = [episodes[int(index)] for index in episode_indices]
    valid_ranges = []
    frames_per_task: dict[tuple[str, ...], int] = {}
    for row in rows:
        task_key = tuple(sorted(row["tasks"]))
        if not task_key:
            raise ValueError(
                "task_sampling=balanced requires at least one task per episode"
            )
        start = int(row["dataset_from_index"])
        stop = int(row["dataset_to_index"]) - int(drop_n_last_frames)
        if stop <= start:
            continue
        valid_ranges.append((start, stop, task_key))
        frames_per_task[task_key] = frames_per_task.get(task_key, 0) + stop - start
    if not valid_ranges:
        raise ValueError("No valid frames remain for task-balanced sampling")

    indices = []
    weights = []
    for start, stop, task_key in valid_ranges:
        indices.extend(range(start, stop))
        weights.extend([1.0 / frames_per_task[task_key]] * (stop - start))

    # WeightedRandomSampler emits positions into its weights array. Map those
    # positions back to absolute dataset indices without loading any images.
    base_sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(indices),
        replacement=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )

    class _AbsoluteIndexSampler:
        def __iter__(self):
            return (indices[position] for position in base_sampler)

        def __len__(self):
            return len(indices)

    logging.info(
        "Using task-balanced sampling across %d tasks and %d valid frames",
        len(frames_per_task),
        len(indices),
    )
    return _AbsoluteIndexSampler()


@torch.no_grad()
def validate(
    policy: MultiTaskDiTPolicy,
    dataloader: DataLoader,
    policy_config: MultiTaskDiTConfig,
    stats_tensors: dict,
    device: str,
    use_amp: bool,
    max_batches: int,
) -> float:
    """Compute mean validation loss over at most ``max_batches`` batches."""
    policy.eval()
    losses = []

    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            batch = move_to_device(batch, device, non_blocking=True)
            normalized_batch = normalize_batch(
                batch,
                policy_config.input_features,
                policy_config.output_features,
                policy_config.normalization_mapping,
                stats_tensors,
            )

            with torch.amp.autocast(device_type=device, enabled=use_amp):
                loss, _ = policy(normalized_batch)

            losses.append(loss.item())
    finally:
        policy.train()

    if not losses:
        return float("nan")
    return float(np.mean(losses))


def train(cfg: TrainConfig):
    _validate_validation_config(cfg.val_dataset_path, cfg.val_ratio)
    if cfg.task_sampling not in {"proportional", "balanced"}:
        raise ValueError(
            "task_sampling must be 'proportional' or 'balanced', got "
            f"{cfg.task_sampling!r}"
        )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if cfg.device == "cpu" or cfg.device.startswith("cpu"):
        logging.warning("=" * 80)
        logging.warning("WARNING: Config device is set to CPU")
        logging.warning("Training will be significantly slower than on GPU")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # An explicit flag also works with credentials cached by `wandb login`.
    # Keep environment-variable auto-detection for backward compatibility.
    use_wandb = cfg.use_wandb or "WANDB_API_KEY" in os.environ
    if use_wandb:
        wandb.init(
            project="multitask-dit-policy",
            name=output_dir.name,
            config=asdict(cfg),
            dir=str(output_dir),
        )

    # Extract repo_id from the final directory name of dataset_path
    dataset_path = Path(cfg.dataset_path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {cfg.dataset_path}\n"
            f"Please ensure the dataset directory exists at the specified path."
        )
    repo_id = dataset_path.name

    # need metadata to get features and stats
    ds_metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_path)

    # Load or create policy
    if cfg.checkpoint_path is not None:
        logging.info(f"Loading policy from checkpoint: {cfg.checkpoint_path}")
        policy = MultiTaskDiTPolicy.load(cfg.checkpoint_path)
        # Use the checkpoint's config for feature setup
        policy_config = policy.config
        logging.info("Policy loaded successfully from checkpoint")
    else:
        policy_config = cfg.policy
        policy = MultiTaskDiTPolicy(policy_config, dataset_metadata=ds_metadata)

    policy_config.device = cfg.device
    policy.to(cfg.device)
    policy.train()

    delta_indices = {}
    # For observation keys
    for key in policy_config.input_features:
        delta_indices[key] = policy_config.observation_delta_indices
    # For action keys
    for key in policy_config.output_features:
        delta_indices[key] = policy_config.action_delta_indices

    delta_timestamps = {key: [i / ds_metadata.fps for i in indices] for key, indices in delta_indices.items()}

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_path,
        delta_timestamps=delta_timestamps,
        video_backend=cfg.video_backend,
    )

    num_episodes = len(dataset.meta.episodes)
    external_val_dataset = None
    if cfg.val_dataset_path is not None:
        val_dataset_path = Path(cfg.val_dataset_path).expanduser()
        if not val_dataset_path.exists():
            raise FileNotFoundError(
                f"Validation dataset path not found: {cfg.val_dataset_path}\n"
                "Please ensure the validation LeRobotDataset directory exists."
            )
        val_repo_id = val_dataset_path.name
        val_metadata = LeRobotDatasetMetadata(repo_id=val_repo_id, root=val_dataset_path)
        feature_keys = set(policy_config.input_features) | set(policy_config.output_features)
        _validate_dataset_compatibility(ds_metadata, val_metadata, feature_keys)
        external_val_dataset = LeRobotDataset(
            repo_id=val_repo_id,
            root=val_dataset_path,
            delta_timestamps=delta_timestamps,
            video_backend=cfg.video_backend,
        )
        train_episode_indices = list(range(num_episodes))
        val_episode_indices = list(range(len(external_val_dataset.meta.episodes)))
        logging.info(
            f"Using external validation dataset: {len(train_episode_indices)} train episodes, "
            f"{len(val_episode_indices)} validation episodes"
        )
    else:
        train_episode_indices, val_episode_indices = _split_episode_indices(num_episodes, cfg.val_ratio, cfg.seed)

    if val_episode_indices and external_val_dataset is None:
        logging.info(
            f"Using episode split: {len(train_episode_indices)} train episodes, "
            f"{len(val_episode_indices)} validation episodes"
        )
    else:
        logging.info("Validation disabled: using all episodes for training")

    if cfg.task_sampling == "balanced":
        sampler = _build_task_balanced_sampler(
            dataset.meta.episodes,
            train_episode_indices,
            drop_n_last_frames=policy_config.drop_n_last_frames,
            seed=cfg.seed,
        )
    else:
        sampler = EpisodeAwareSampler(
            _to_episode_subset(dataset.meta.episodes["dataset_from_index"], train_episode_indices),
            _to_episode_subset(dataset.meta.episodes["dataset_to_index"], train_episode_indices),
            drop_n_last_frames=policy_config.drop_n_last_frames,
            shuffle=True,
            seed=cfg.seed,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True if cfg.device == "cuda" else False,
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    val_dataloader = None
    if val_episode_indices and cfg.val_freq > 0 and cfg.val_batches > 0:
        val_dataset = external_val_dataset if external_val_dataset is not None else dataset
        val_sampler = EpisodeAwareSampler(
            _to_episode_subset(val_dataset.meta.episodes["dataset_from_index"], val_episode_indices),
            _to_episode_subset(val_dataset.meta.episodes["dataset_to_index"], val_episode_indices),
            drop_n_last_frames=policy_config.drop_n_last_frames,
            shuffle=False,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.device.startswith("cuda"),
            persistent_workers=cfg.num_workers > 0,
            drop_last=False,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )

    probe_dataloader = None
    if cfg.probe_dataset_path is not None:
        probe_dataset_path = Path(cfg.probe_dataset_path).expanduser()
        if not probe_dataset_path.exists():
            raise FileNotFoundError(
                f"Probe dataset path not found: {cfg.probe_dataset_path}\n"
                "Please ensure the probe LeRobotDataset directory exists."
            )
        probe_repo_id = probe_dataset_path.name
        probe_metadata = LeRobotDatasetMetadata(repo_id=probe_repo_id, root=probe_dataset_path)
        feature_keys = set(policy_config.input_features) | set(policy_config.output_features)
        _validate_dataset_compatibility(ds_metadata, probe_metadata, feature_keys)
        probe_dataset = LeRobotDataset(
            repo_id=probe_repo_id,
            root=probe_dataset_path,
            delta_timestamps=delta_timestamps,
            video_backend=cfg.video_backend,
        )
        probe_indices = list(range(len(probe_dataset.meta.episodes)))
        probe_sampler = EpisodeAwareSampler(
            _to_episode_subset(probe_dataset.meta.episodes["dataset_from_index"], probe_indices),
            _to_episode_subset(probe_dataset.meta.episodes["dataset_to_index"], probe_indices),
            drop_n_last_frames=policy_config.drop_n_last_frames,
            shuffle=False,
        )
        probe_dataloader = DataLoader(
            probe_dataset,
            batch_size=cfg.batch_size,
            sampler=probe_sampler,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.device.startswith("cuda"),
            persistent_workers=cfg.num_workers > 0,
            drop_last=False,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )
        logging.info(
            "Using read-only probe dataset: %d episodes from %s",
            len(probe_indices),
            probe_dataset_path,
        )

    optimizer_config = policy_config.get_optimizer_preset()
    optimizer_params = policy.get_optim_params()
    optimizer = torch.optim.Adam(
        optimizer_params,
        lr=optimizer_config.lr,
        betas=optimizer_config.betas,
        eps=optimizer_config.eps,
        weight_decay=optimizer_config.weight_decay,
    )

    use_amp = cfg.use_amp and cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if use_amp:
        logging.info("Using Automatic Mixed Precision (AMP) for training")

    stats_tensors = {}
    for key, stat in ds_metadata.stats.items():
        stats_tensors[key] = {k: torch.tensor(v, device=cfg.device, dtype=torch.float32) for k, v in stat.items()}

    step = 0
    progress_bar = tqdm(total=cfg.train_steps)

    # infinite iterator over dataloader (cycles automatically)
    dataloader_iter = _cycle(dataloader)

    while step < cfg.train_steps:
        batch = next(dataloader_iter)
        batch = move_to_device(batch, cfg.device, non_blocking=True)

        normalized_batch = normalize_batch(
            batch,
            policy_config.input_features,
            policy_config.output_features,
            policy_config.normalization_mapping,
            stats_tensors,
        )

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=cfg.device, enabled=use_amp):
            loss, _ = policy(normalized_batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # unscale for gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        grad_norm_value = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
        scaler.step(optimizer)
        scaler.update()

        step += 1

        if step % cfg.save_freq == 0:
            save_path = output_dir / f"checkpoint_{step}"
            save_policy(policy, save_path, ds_metadata.stats)
            logging.info(f"Saved checkpoint to {save_path}")
            if use_wandb:
                wandb.log({"checkpoint_step": step}, step=step)

        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss.item())

        if step % cfg.log_freq == 0:
            logging.info(f"Step {step}: Loss = {loss.item():.6f}, Grad Norm = {grad_norm_value:.6f}")
            if use_wandb:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "train/loss": loss.item(),
                        "train/grad_norm": grad_norm_value,
                        "step": step,
                    },
                    step=step,
                )

        if val_dataloader is not None and step % cfg.val_freq == 0:
            val_loss = validate(
                policy,
                val_dataloader,
                policy_config,
                stats_tensors,
                cfg.device,
                use_amp,
                cfg.val_batches,
            )
            logging.info(f"Step {step}: Validation Loss = {val_loss:.6f}")
            if use_wandb:
                wandb.log({"val/loss": val_loss}, step=step)

        if (
            probe_dataloader is not None
            and cfg.probe_freq > 0
            and cfg.probe_batches > 0
            and step % cfg.probe_freq == 0
        ):
            probe_loss = validate(
                policy,
                probe_dataloader,
                policy_config,
                stats_tensors,
                cfg.device,
                use_amp,
                cfg.probe_batches,
            )
            logging.info(f"Step {step}: Probe Loss = {probe_loss:.6f}")
            if use_wandb:
                wandb.log({"probe/loss": probe_loss}, step=step)

    final_dir = output_dir / "final_model"
    save_policy(policy, final_dir, ds_metadata.stats)
    logging.info("Training finished.")
    if use_wandb:
        wandb.finish()


@draccus.wrap()
def main(cfg: TrainConfig):
    train(cfg)


if __name__ == "__main__":
    main()
