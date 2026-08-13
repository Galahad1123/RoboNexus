"""Tests for the CALVIN state-inspection backend adapter."""

import unittest

import numpy as np

from backends.calvin.verification.state_inspector import CalvinStateInspector
from core.legacy.inspection import InspectionStatus


def observation(slider=0.0, drawer=0.0, lightbulb=0.0, led=0.0):
    scene = np.zeros(24, dtype=np.float32)
    scene[0] = slider
    scene[1] = drawer
    scene[4] = lightbulb
    scene[5] = led
    return {"scene_obs": scene}


class CalvinStateInspectorTest(unittest.TestCase):
    def setUp(self):
        self.inspector = CalvinStateInspector()

    def test_absolute_facility_tasks_are_skipped_only_when_satisfied(self):
        state = observation(slider=0.28, drawer=0.22, lightbulb=1, led=0)
        expected = {
            "open_drawer": InspectionStatus.SATISFIED,
            "close_drawer": InspectionStatus.READY,
            "move_slider_left": InspectionStatus.SATISFIED,
            "move_slider_right": InspectionStatus.READY,
            "turn_on_lightbulb": InspectionStatus.SATISFIED,
            "turn_off_lightbulb": InspectionStatus.READY,
            "turn_on_led": InspectionStatus.READY,
            "turn_off_led": InspectionStatus.SATISFIED,
        }
        for skill_id, status in expected.items():
            with self.subTest(skill_id=skill_id):
                self.assertEqual(self.inspector.inspect(skill_id, state).status, status)

    def test_drawer_operations_report_invalid_closed_precondition(self):
        state = observation(drawer=0.0)
        for skill_id in (
            "lift_blue_block_drawer",
            "lift_pink_block_drawer",
            "lift_red_block_drawer",
            "place_in_drawer",
            "push_into_drawer",
        ):
            with self.subTest(skill_id=skill_id):
                result = self.inspector.inspect(skill_id, state)
                self.assertEqual(
                    result.status, InspectionStatus.INVALID_PRECONDITION
                )
                self.assertEqual(result.reason, "drawer_must_be_open")

    def test_open_drawer_satisfies_drawer_preconditions(self):
        result = self.inspector.inspect(
            "place_in_drawer", observation(drawer=0.22)
        )
        self.assertEqual(result.status, InspectionStatus.READY)
        self.assertEqual(result.reason, "drawer_precondition_satisfied")

    def test_uninspected_object_skill_remains_ready(self):
        result = self.inspector.inspect(
            "rotate_red_block_left", observation(drawer=0.0)
        )
        self.assertEqual(result.status, InspectionStatus.READY)
        self.assertEqual(result.reason, "no_fixed_facility_precondition")

    def test_missing_or_malformed_scene_state_is_unknown(self):
        self.assertEqual(
            self.inspector.inspect("open_drawer", {}).status,
            InspectionStatus.UNKNOWN,
        )
        self.assertEqual(
            self.inspector.inspect(
                "open_drawer", {"scene_obs": np.zeros(6)}
            ).status,
            InspectionStatus.UNKNOWN,
        )

    def test_scene_info_dictionary_is_also_supported(self):
        state = {
            "scene_info": {
                "doors": {
                    "base__slide": {"current_state": [0.28]},
                    "base__drawer": {"current_state": [0.22]},
                },
                "lights": {
                    "lightbulb": {"logical_state": 1},
                    "led": {"logical_state": 0},
                },
            }
        }
        snapshot = self.inspector.snapshot(state)
        self.assertEqual(snapshot["drawer"], "open")
        self.assertEqual(snapshot["slider"], "left")
        self.assertEqual(snapshot["lightbulb"], "on")
        self.assertEqual(snapshot["led"], "off")


if __name__ == "__main__":
    unittest.main()
