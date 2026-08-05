"""Absolute CALVIN fixed-facility state inspection."""

from dataclasses import dataclass

import numpy as np

from core.legacy.inspection import InspectionStatus, SkillInspection


ABSOLUTE_TARGETS = {
    "open_drawer": ("drawer", "open"),
    "close_drawer": ("drawer", "closed"),
    "move_slider_left": ("slider", "left"),
    "move_slider_right": ("slider", "right"),
    "turn_on_lightbulb": ("lightbulb", "on"),
    "turn_off_lightbulb": ("lightbulb", "off"),
    "turn_on_led": ("led", "on"),
    "turn_off_led": ("led", "off"),
}

DRAWER_OPEN_REQUIRED = frozenset(
    [
        "lift_blue_block_drawer",
        "lift_pink_block_drawer",
        "lift_red_block_drawer",
        "place_in_drawer",
        "push_into_drawer",
    ]
)


@dataclass(frozen=True)
class CalvinFacilityThresholds:
    drawer_closed_max: float = 0.05
    drawer_open_min: float = 0.10
    slider_right_max: float = 0.10
    slider_left_min: float = 0.20
    light_on_min: float = 0.5


class CalvinStateInspector:
    """Inspect only absolute states that are stable in CALVIN task-D."""

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or CalvinFacilityThresholds()

    def inspect(self, skill_id, observation):
        try:
            snapshot = self.snapshot(observation)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            return SkillInspection(
                InspectionStatus.UNKNOWN,
                "facility_state_unavailable",
                {"error": "{}: {}".format(type(exc).__name__, exc)},
            )

        if skill_id in ABSOLUTE_TARGETS:
            facility, desired = ABSOLUTE_TARGETS[skill_id]
            actual = snapshot[facility]
            evidence = {
                "facility": facility,
                "actual": actual,
                "desired": desired,
                "raw": snapshot["{}_raw".format(facility)],
            }
            if actual == desired:
                return SkillInspection(
                    InspectionStatus.SATISFIED,
                    "absolute_state_already_satisfied",
                    evidence,
                )
            return SkillInspection(
                InspectionStatus.READY,
                "absolute_state_change_required",
                evidence,
            )

        if skill_id in DRAWER_OPEN_REQUIRED:
            actual = snapshot["drawer"]
            evidence = {
                "facility": "drawer",
                "actual": actual,
                "required": "open",
                "raw": snapshot["drawer_raw"],
            }
            if actual != "open":
                return SkillInspection(
                    InspectionStatus.INVALID_PRECONDITION,
                    "drawer_must_be_open",
                    evidence,
                )
            return SkillInspection(
                InspectionStatus.READY,
                "drawer_precondition_satisfied",
                evidence,
            )

        return SkillInspection(
            InspectionStatus.READY,
            "no_fixed_facility_precondition",
            {},
        )

    def snapshot(self, observation):
        values = _facility_values(observation)
        thresholds = self.thresholds
        drawer = (
            "closed"
            if values["drawer"] <= thresholds.drawer_closed_max
            else "open"
            if values["drawer"] >= thresholds.drawer_open_min
            else "intermediate"
        )
        slider = (
            "right"
            if values["slider"] <= thresholds.slider_right_max
            else "left"
            if values["slider"] >= thresholds.slider_left_min
            else "intermediate"
        )
        return {
            "drawer": drawer,
            "drawer_raw": values["drawer"],
            "slider": slider,
            "slider_raw": values["slider"],
            "lightbulb": (
                "on" if values["lightbulb"] >= thresholds.light_on_min else "off"
            ),
            "lightbulb_raw": values["lightbulb"],
            "led": "on" if values["led"] >= thresholds.light_on_min else "off",
            "led_raw": values["led"],
        }


def _facility_values(observation):
    if not isinstance(observation, dict):
        raise TypeError("observation must be a mapping")
    if "scene_obs" in observation:
        scene = np.asarray(observation["scene_obs"], dtype=np.float64).reshape(-1)
        if scene.size != 24:
            raise ValueError("scene_obs must contain 24 values")
        if not np.all(np.isfinite(scene[:6])):
            raise ValueError("scene_obs fixed-facility values must be finite")
        return {
            "slider": float(scene[0]),
            "drawer": float(scene[1]),
            "lightbulb": float(scene[4]),
            "led": float(scene[5]),
        }
    scene_info = observation.get("scene_info")
    info = observation.get("info")
    if scene_info is None and isinstance(info, dict):
        scene_info = info.get("scene_info")
    if not isinstance(scene_info, dict):
        raise KeyError("scene_obs or scene_info is required")
    return {
        "slider": _scalar(
            scene_info["doors"]["base__slide"]["current_state"]
        ),
        "drawer": _scalar(
            scene_info["doors"]["base__drawer"]["current_state"]
        ),
        "lightbulb": _scalar(
            scene_info["lights"]["lightbulb"]["logical_state"]
        ),
        "led": _scalar(scene_info["lights"]["led"]["logical_state"]),
    }


def _scalar(value):
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 1 or not np.isfinite(array[0]):
        raise ValueError("facility joint state must be one finite value")
    return float(array[0])
