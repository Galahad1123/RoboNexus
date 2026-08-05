"""CALVIN implementation of the inspect-scene environment port."""

import numpy as np

from core.domain import SceneCapture
from core.ports import EnvironmentPort, EnvironmentPortError

from ..verification.state_inspector import CalvinStateInspector


class CalvinSceneObserver(EnvironmentPort):
    def __init__(self, adapter, state_inspector=None):
        self.adapter = adapter
        self.state_inspector = state_inspector or CalvinStateInspector()

    @property
    def environment_steps(self):
        return int(getattr(self.adapter, "environment_steps", 0))

    def capture_scene(self):
        observation = getattr(self.adapter, "current_observation", None)
        if not isinstance(observation, dict):
            raise EnvironmentPortError(
                "CALVIN environment has no current observation; reset it first"
            )
        try:
            rgb = observation["rgb_obs"]
            static = np.asarray(rgb["rgb_static"], dtype=np.uint8).copy()
            wrist = np.asarray(rgb["rgb_gripper"], dtype=np.uint8).copy()
        except (KeyError, TypeError, ValueError) as exc:
            raise EnvironmentPortError(
                "invalid CALVIN camera observation: {}".format(exc)
            )
        result = SceneCapture(
            static_rgb=static,
            wrist_rgb=wrist,
            environment_steps=self.environment_steps,
            robot_summary=self._robot_summary(observation),
            facility_summary=self._facility_summary(observation),
        )
        result.validate()
        return result

    def _facility_summary(self, observation):
        try:
            snapshot = self.state_inspector.snapshot(observation)
        except (KeyError, TypeError, ValueError, IndexError):
            return {name: "unknown" for name in ("drawer", "slider", "lightbulb", "led")}
        return {
            "drawer": snapshot["drawer"],
            "slider": snapshot["slider"],
            "lightbulb": snapshot["lightbulb"],
            "led": snapshot["led"],
        }

    @staticmethod
    def _robot_summary(observation):
        try:
            robot_obs = np.asarray(observation["robot_obs"]).reshape(-1)
            gripper = "open" if float(robot_obs[14]) > 0 else "closed"
        except (KeyError, TypeError, ValueError, IndexError):
            gripper = "unknown"
        return {"gripper": gripper, "holding_object": "unknown"}
