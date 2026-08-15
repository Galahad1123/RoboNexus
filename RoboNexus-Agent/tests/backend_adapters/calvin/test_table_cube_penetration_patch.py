import unittest
from types import SimpleNamespace

from backends.calvin.patches import TableCubePenetrationDetector


def info(z=0.46, support_link=-1, x=0.1, y=-0.1):
    return {
        "robot_info": {"uid": 99},
        "scene_info": {
            "fixed_objects": {
                "table": {
                    "uid": 10,
                    "links": {"drawer_link": 3, "plank_link": 4},
                }
            },
            "movable_objects": {
                "block_red": {
                    "uid": 20,
                    "current_pos": (x, y, z),
                    "current_orn": (0.0, 0.0, 0.0, 1.0),
                    "current_lin_vel": (0.0, 0.0, 0.0),
                    "current_ang_vel": (0.0, 0.0, 0.0),
                    "contacts": [(0, 0, 10, 0, support_link)],
                }
            },
        },
    }


class FakeBullet:
    def __init__(self):
        self.pose = None
        self.velocity = None
        self.steps = 0

    def resetBasePositionAndOrientation(self, uid, position, orientation, physicsClientId):
        self.pose = (uid, tuple(position), tuple(orientation), physicsClientId)

    def resetBaseVelocity(self, uid, linearVelocity, angularVelocity, physicsClientId):
        self.velocity = (uid, tuple(linearVelocity), tuple(angularVelocity), physicsClientId)

    def stepSimulation(self, physicsClientId):
        self.steps += 1


class FakeAdapter:
    def __init__(self):
        self.current_info = info()
        self.env = SimpleNamespace(p=FakeBullet(), cid=7, get_obs=lambda: {"refreshed": True})
        self.current_observation = None

    def get_info(self):
        return self.current_info


class TableCubePenetrationPatchTest(unittest.TestCase):
    def detector(self, adapter, skill="push_red_block_left", support=-1):
        initial = info(support_link=support)
        adapter.current_info = initial
        grounding = SimpleNamespace(
            role=SimpleNamespace(value="source"), semantic_id="block_red"
        )
        return TableCubePenetrationDetector.for_attempt(
            adapter, skill, (grounding,), initial
        )

    def test_normal_motion_lift_and_small_contact_depth_are_untouched(self):
        adapter = FakeAdapter()
        detector = self.detector(adapter)
        self.assertIsNone(detector.observe(info(0.46)))
        self.assertIsNone(detector.observe(info(0.455)))
        self.assertIsNone(detector.observe(info(0.47)))
        self.assertIsNone(adapter.env.p.pose)

    def test_two_deep_samples_trigger_and_restore_last_valid_pose(self):
        adapter = FakeAdapter()
        detector = self.detector(adapter)
        self.assertIsNone(detector.observe(info(0.455)))
        self.assertIsNone(detector.observe(info(0.450)))
        anomaly = detector.observe(info(0.449))
        self.assertEqual(anomaly["type"], "table_cube_penetration")
        adapter.current_info = info(0.449)
        recovered = detector.recover()
        self.assertTrue(recovered["performed"])
        self.assertAlmostEqual(adapter.env.p.pose[1][2], 0.455)
        self.assertEqual(adapter.env.p.velocity[1], (0.0, 0.0, 0.0))
        self.assertEqual(adapter.env.p.velocity[2], (0.0, 0.0, 0.0))
        self.assertEqual(adapter.env.p.steps, 16)
        self.assertEqual(adapter.current_observation, {"refreshed": True})

    def test_non_table_and_non_targeted_skills_are_disabled(self):
        adapter = FakeAdapter()
        drawer_detector = self.detector(adapter, support=3)
        place_detector = self.detector(adapter, skill="place_in_drawer")
        self.assertFalse(drawer_detector.enabled)
        self.assertFalse(place_detector.enabled)
        self.assertIsNone(drawer_detector.observe(info(0.40)))
        self.assertIsNone(place_detector.observe(info(0.40)))

    def test_cube_falling_beyond_table_edge_is_not_rewound_as_penetration(self):
        adapter = FakeAdapter()
        detector = self.detector(adapter)
        self.assertIsNone(detector.observe(info(0.43, x=-0.37, y=-0.20)))
        self.assertIsNone(detector.observe(info(0.38, x=-0.38, y=-0.21)))
        self.assertFalse(detector.enabled)
        self.assertEqual(detector.recover()["reason"], "no_penetration_detected")
        self.assertIsNone(adapter.env.p.pose)


if __name__ == "__main__":
    unittest.main()
