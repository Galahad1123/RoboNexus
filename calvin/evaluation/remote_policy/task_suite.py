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
        self.dataset_split = self.source_dataset / self.split
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

    def episodes(self, task, count=None, seed=0):
        if task not in self.tasks:
            raise KeyError("Unknown suite task: {}".format(task))
        task_config = self.tasks[task]
        rows = [
            row
            for row in self.manifest["splits"][self.split]["segments"]
            if row["calvin_task"] == task
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
