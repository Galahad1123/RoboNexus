#!/usr/bin/env python

"""Distributed training entry point for Multi-Task DiT using Accelerate.

``batch_size`` is the per-process (normally per-GPU) batch size. Therefore the
effective global batch size is ``batch_size * num_processes``. This module is
kept separate from :mod:`multitask_dit_policy.train` so the established
single-GPU entry point remains unchanged.
"""

import logging
import os
import random
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from multitask_dit_policy.model.model import MultiTaskDiTPolicy
from multitask_dit_policy.train import (
    TrainConfig,
    _build_task_balanced_sampler,
    _cycle,
    _split_episode_indices,
    _to_episode_subset,
    _validate_dataset_compatibility,
    _validate_validation_config,
)
from multitask_dit_policy.utils.utils import move_to_device, normalize_batch, save_policy

warnings.filterwarnings("ignore", message=".*Field.*attribute.*repr.*")
warnings.filterwarnings("ignore", message=".*Field.*attribute.*frozen.*")
warnings.filterwarnings("ignore", module="pydantic._internal._generate_schema")

import draccus  # noqa: E402


def effective_batch_size(per_device_batch_size: int, num_processes: int) -> int:
    """Return the effective batch size without gradient accumulation."""
    if per_device_batch_size < 1 or num_processes < 1:
        raise ValueError("per_device_batch_size and num_processes must both be positive")
    return per_device_batch_size * num_processes


@torch.no_grad()
def validate_distributed(
    accelerator: Accelerator,
    policy: torch.nn.Module,
    dataloader: DataLoader,
    policy_config,
    stats_tensors: dict,
    max_batches: int,
) -> float:
    """Compute a sample-weighted validation loss across all processes."""
    policy.eval()
    loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    sample_count = torch.zeros((), device=accelerator.device, dtype=torch.float64)

    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            batch = move_to_device(batch, accelerator.device, non_blocking=True)
            normalized_batch = normalize_batch(
                batch,
                policy_config.input_features,
                policy_config.output_features,
                policy_config.normalization_mapping,
                stats_tensors,
            )
            with accelerator.autocast():
                loss, _ = policy(normalized_batch)

            local_batch_size = next(value for value in batch.values() if isinstance(value, torch.Tensor)).shape[0]
            loss_sum += loss.detach().double() * local_batch_size
            sample_count += local_batch_size
    finally:
        policy.train()

    totals = torch.stack((loss_sum, sample_count))
    totals = accelerator.reduce(totals, reduction="sum")
    if totals[1].item() == 0:
        return float("nan")
    return (totals[0] / totals[1]).item()


def _save_on_main_process(
    accelerator: Accelerator,
    policy: torch.nn.Module,
    save_path: Path,
    dataset_stats: dict,
) -> None:
    """Synchronize workers and save one load-compatible policy checkpoint."""
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_policy(accelerator.unwrap_model(policy), save_path, dataset_stats)
    accelerator.wait_for_everyone()


def train(cfg: TrainConfig) -> None:
    _validate_validation_config(cfg.val_dataset_path, cfg.val_ratio)
    if cfg.task_sampling not in {"proportional", "balanced"}:
        raise ValueError(f"task_sampling must be 'proportional' or 'balanced', got {cfg.task_sampling!r}")

    mixed_precision = "fp16" if cfg.use_amp else "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    set_seed(cfg.seed, device_specific=True)
    random.seed(cfg.seed + accelerator.process_index)
    np.random.seed(cfg.seed + accelerator.process_index)

    device = accelerator.device
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    output_dir = Path(cfg.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    use_wandb = cfg.use_wandb or "WANDB_API_KEY" in os.environ
    if use_wandb and accelerator.is_main_process:
        wandb.init(
            project="multitask-dit-policy",
            name=output_dir.name,
            config=asdict(cfg),
            dir=str(output_dir),
        )

    dataset_path = Path(cfg.dataset_path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {cfg.dataset_path}")
    repo_id = dataset_path.name
    ds_metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_path)

    if cfg.checkpoint_path is not None:
        accelerator.print(f"Loading policy from checkpoint: {cfg.checkpoint_path}")
        policy = MultiTaskDiTPolicy.load(cfg.checkpoint_path)
        policy_config = policy.config
    else:
        policy_config = cfg.policy
        policy = MultiTaskDiTPolicy(policy_config, dataset_metadata=ds_metadata)

    accelerator.print(f"Task conditioning: {policy.task_conditioning_summary()}")
    if policy.task_conditioning_type == "task_id":
        id_key = policy.config.observation_encoder.text.id_feature_key
        if id_key not in ds_metadata.features:
            raise ValueError(
                f"Task-ID conditioning requires dataset feature {id_key!r}; "
                f"available features: {sorted(ds_metadata.features)}"
            )

    policy_config.device = str(device)
    policy.to(device)
    policy.train()

    delta_indices = dict.fromkeys(policy_config.input_features, policy_config.observation_delta_indices)
    delta_indices.update(dict.fromkeys(policy_config.output_features, policy_config.action_delta_indices))
    delta_timestamps = {key: [index / ds_metadata.fps for index in indices] for key, indices in delta_indices.items()}
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
            raise FileNotFoundError(f"Validation dataset path not found: {cfg.val_dataset_path}")
        val_repo_id = val_dataset_path.name
        val_metadata = LeRobotDatasetMetadata(repo_id=val_repo_id, root=val_dataset_path)
        if policy.task_conditioning_type == "task_id":
            id_key = policy.config.observation_encoder.text.id_feature_key
            if id_key not in val_metadata.features:
                raise ValueError(
                    f"Task-ID conditioning requires validation feature {id_key!r}"
                )
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
    else:
        train_episode_indices, val_episode_indices = _split_episode_indices(num_episodes, cfg.val_ratio, cfg.seed)

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
        pin_memory=device.type == "cuda",
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
            pin_memory=device.type == "cuda",
            persistent_workers=cfg.num_workers > 0,
            drop_last=False,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )

    optimizer_config = policy_config.get_optimizer_preset()
    optimizer = torch.optim.Adam(
        policy.get_optim_params(),
        lr=optimizer_config.lr,
        betas=optimizer_config.betas,
        eps=optimizer_config.eps,
        weight_decay=optimizer_config.weight_decay,
    )

    if val_dataloader is None:
        policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
    else:
        policy, optimizer, dataloader, val_dataloader = accelerator.prepare(
            policy, optimizer, dataloader, val_dataloader
        )

    stats_tensors = {
        key: {name: torch.as_tensor(value, device=device, dtype=torch.float32) for name, value in stat.items()}
        for key, stat in ds_metadata.stats.items()
    }

    global_batch = effective_batch_size(cfg.batch_size, accelerator.num_processes)
    accelerator.print(
        f"Starting distributed training: processes={accelerator.num_processes}, "
        f"per_device_batch={cfg.batch_size}, global_batch={global_batch}, device={device}, "
        f"mixed_precision={mixed_precision}"
    )

    progress_bar = tqdm(
        total=cfg.train_steps,
        disable=not accelerator.is_local_main_process,
    )
    dataloader_iter = _cycle(dataloader)

    for step in range(1, cfg.train_steps + 1):
        batch = next(dataloader_iter)
        batch = move_to_device(batch, device, non_blocking=True)
        normalized_batch = normalize_batch(
            batch,
            policy_config.input_features,
            policy_config.output_features,
            policy_config.normalization_mapping,
            stats_tensors,
        )

        optimizer.zero_grad()
        with accelerator.autocast():
            loss, _ = policy(normalized_batch)
        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        reduced_loss = accelerator.reduce(loss.detach(), reduction="mean")
        reduced_grad_norm = accelerator.reduce(torch.as_tensor(grad_norm, device=device).detach(), reduction="mean")
        progress_bar.update(1)
        progress_bar.set_postfix(loss=reduced_loss.item())

        if step % cfg.log_freq == 0 and accelerator.is_main_process:
            logging.info(
                "Step %d: Loss = %.6f, Grad Norm = %.6f",
                step,
                reduced_loss.item(),
                reduced_grad_norm.item(),
            )
            if use_wandb:
                wandb.log(
                    {
                        "loss": reduced_loss.item(),
                        "train/loss": reduced_loss.item(),
                        "train/grad_norm": reduced_grad_norm.item(),
                        "train/global_batch_size": global_batch,
                        "step": step,
                    },
                    step=step,
                )

        if step % cfg.save_freq == 0:
            save_path = output_dir / f"checkpoint_{step}"
            _save_on_main_process(accelerator, policy, save_path, ds_metadata.stats)
            if accelerator.is_main_process:
                logging.info("Saved checkpoint to %s", save_path)
                if use_wandb:
                    wandb.log({"checkpoint_step": step}, step=step)

        if val_dataloader is not None and step % cfg.val_freq == 0:
            val_loss = validate_distributed(
                accelerator,
                policy,
                val_dataloader,
                policy_config,
                stats_tensors,
                cfg.val_batches,
            )
            if accelerator.is_main_process:
                logging.info("Step %d: Validation Loss = %.6f", step, val_loss)
                if use_wandb:
                    wandb.log({"val/loss": val_loss}, step=step)

    progress_bar.close()
    _save_on_main_process(accelerator, policy, output_dir / "final_model", ds_metadata.stats)
    if accelerator.is_main_process:
        logging.info("Training finished.")
        if use_wandb:
            wandb.finish()
    accelerator.end_training()


@draccus.wrap()
def main(cfg: TrainConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
