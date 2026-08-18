"""Configuration-driven CALVIN evaluation suites."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml


@dataclass
class EpisodeSpec:
    episode_id: str
    task: str
    instruction: str
    oracle_task: str
    source_frame: Path
    max_steps: int
    settle_steps: int
    episode_variant: Optional[str] = None
    success_guard: Optional[dict] = None
    mask_roles: Optional[Dict[str, List[dict]]] = None
    retry: Optional[dict] = None

    def load_initial_state(self):
        with np.load(str(self.source_frame)) as frame:
            return (
                frame["robot_obs"].copy(),
                frame["scene_obs"].copy(),
            )


class ConfiguredCalvinSuite:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.workspace = self._find_workspace(self.config_path)
        self.config = yaml.safe_load(self.config_path.read_text())
        self.manifest_path = self._resolve(self.config["manifest"])
        self.manifest = json.loads(self.manifest_path.read_text())
        self.source_dataset = self._resolve(
            self.config.get("source_dataset", self.manifest["source_dataset"])
        )
        self.split = self.config.get("split", "validation")
        if self.split not in self.manifest["splits"]:
            raise ValueError(
                "Suite split {!r} is not present in manifest {}".format(
                    self.split, self.manifest_path
                )
            )
        split_config = self.manifest["splits"][self.split]
        # A logical dataset split may be derived from another physical CALVIN
        # directory. For example, ABC manifests expose ``d_validation`` while
        # its NPZ files and simulator config live in the official
        # ``validation`` directory.
        self.source_split = split_config.get("source_split", self.split)
        self.dataset_split = self.source_dataset / self.source_split
        self.tasks = {item["task"]: item for item in self.config["tasks"]}
        unknown = set(self.tasks) - {
            row["calvin_task"] for row in self.manifest["splits"][self.split]["segments"]
        }
        if unknown:
            raise ValueError(
                "Suite tasks have no retained manifest segments: {}".format(sorted(unknown))
            )

    def task_names(self):
        return list(self.tasks)

    def episode_variant_names(self):
        return sorted(
            {
                row["episode_variant"]
                for row in self.manifest["splits"][self.split]["segments"]
                if row.get("episode_variant")
            }
        )

    def episodes(self, task, count=None, seed=0, episode_variants=None):
        if task not in self.tasks:
            raise KeyError("Unknown suite task: {}".format(task))
        task_config = self.tasks[task]
        rows = [
            row
            for row in self.manifest["splits"][self.split]["segments"]
            if row["calvin_task"] == task
        ]
        incompatible_sources = sorted(
            {
                row.get("source_split")
                for row in rows
                if row.get("source_split", self.source_split) != self.source_split
            }
        )
        if incompatible_sources:
            raise ValueError(
                "Manifest rows for {} reference source splits {}, expected {}".format(
                    task, incompatible_sources, self.source_split
                )
            )
        if episode_variants:
            rows = [
                row for row in rows if row.get("episode_variant") in episode_variants
            ]
        rows.sort(key=lambda row: row["segment_id"])
        if count is not None and count < len(rows):
            rng = np.random.RandomState(seed)
            indices = sorted(rng.choice(len(rows), size=count, replace=False).tolist())
            rows = [rows[index] for index in indices]

        return [
            EpisodeSpec(
                episode_id=row["segment_id"],
                task=task,
                instruction=task_config.get("instruction", row["instruction"]),
                oracle_task=task_config.get("oracle_task", task),
                source_frame=self.dataset_split
                / "episode_{:07d}.npz".format(int(row["start"])),
                max_steps=int(task_config.get("max_steps", self.config["defaults"]["max_steps"])),
                settle_steps=int(
                    task_config.get("settle_steps", self.config["defaults"].get("settle_steps", 0))
                ),
                episode_variant=row.get("episode_variant"),
                success_guard=task_config.get("success_guard"),
                # Specialist manifests may resolve a different destination for
                # every demonstration (for example the demonstrated stack
                # target). A suite-level value remains an explicit override.
                mask_roles=(
                    task_config["mask_roles"]
                    if "mask_roles" in task_config
                    else row.get("mask_roles")
                ),
                retry=task_config.get("retry"),
            )
            for row in rows
        ]

    def create_task_oracle(self):
        import hydra
        from omegaconf import OmegaConf

        oracle_config = OmegaConf.load(self._resolve(self.config["oracle_config"]))
        if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            hydra.initialize(".")
        return hydra.utils.instantiate(oracle_config)

    def _resolve(self, value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.workspace / path

    @staticmethod
    def _find_workspace(path):
        for parent in (path,) + tuple(path.parents):
            if (parent / "ARCHITECTURE.md").is_file():
                return parent
        raise FileNotFoundError("Could not locate RoboNexus root from {}".format(path))
