"""CALVIN implementation of the backend-neutral approach controller port."""

import numpy as np

from core.domain import ApproachExecutionResult
from core.ports import (
    ApproachControllerPort,
    ApproachControllerPortError,
    ApproachRejected,
    ApproachUnavailable,
)

from ..environment.scene_observer import CalvinSceneObserver
from ..grounding.semantic_vocabulary import load_semantic_vocabulary
from .motion_controller import CalvinCartesianApproachController
from .target_localizer import CalvinOracleTargetLocalizer


class CalvinApproachController(ApproachControllerPort):
    """Resolve a current grounding and conservatively approach it in XY."""

    def __init__(
        self,
        adapter,
        vocabulary=None,
        vocabulary_path=None,
        hover_height_m=0.12,
        z_tolerance_m=0.015,
        workspace_bounds=None,
    ):
        self.adapter = adapter
        self.vocabulary = vocabulary or load_semantic_vocabulary(vocabulary_path)
        self.localizer = CalvinOracleTargetLocalizer(adapter.env)
        self.motion = CalvinCartesianApproachController(adapter)
        self.hover_height_m = float(hover_height_m)
        self.z_tolerance_m = float(z_tolerance_m)
        self.workspace_bounds = workspace_bounds

    def approach(self, grounding, xy_threshold_m, step_limit):
        observation = getattr(self.adapter, "current_observation", None)
        if not isinstance(observation, dict):
            raise ApproachUnavailable(
                "CALVIN environment has no current observation; reset it first"
            )
        try:
            target_id, target_position, _ = self.localizer.locate_grounding(
                grounding, self.vocabulary
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ApproachRejected(
                "cannot localize grounding {!r}: {}".format(
                    grounding.grounding_id, exc
                )
            )
        initial = float(
            np.linalg.norm(
                np.asarray(observation["robot_obs"][:2], dtype=float)
                - np.asarray(target_position[:2], dtype=float)
            )
        )
        config = {
            "xy_threshold_m": float(xy_threshold_m),
            "hover_height_m": self.hover_height_m,
            "z_tolerance_m": self.z_tolerance_m,
            "max_motion_steps": int(step_limit),
        }
        if self.workspace_bounds is not None:
            config["workspace_bounds"] = self.workspace_bounds
        try:
            result = self.motion.execute(
                observation, target_id, target_position, config
            )
            capture = CalvinSceneObserver(self.adapter).capture_scene()
        except ApproachControllerPortError:
            raise
        except Exception as exc:
            raise ApproachControllerPortError(
                "CALVIN approach failed: {}: {}".format(type(exc).__name__, exc)
            )
        return ApproachExecutionResult(
            success=bool(result.success),
            reason=result.reason,
            steps=int(result.steps),
            initial_xy_distance_m=initial,
            final_xy_distance_m=float(result.final_xy_distance_m),
            threshold_m=float(xy_threshold_m),
            scene_capture=capture,
            metadata={
                "target_id": result.target_id,
                "target_position": np.asarray(result.target_position).tolist(),
                "actions": list(result.actions or ()),
            },
        )
