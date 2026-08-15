import unittest

from backends.calvin.environment.object_locator import CalvinOracleObjectLocator


class Adapter:
    def get_info(self):
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
                    "block_red": {"contacts": [(0, 0, 10, 0, -1)]},
                    "block_blue": {"contacts": [(0, 0, 10, 0, 4)]},
                    "block_pink": {"contacts": [(0, 0, 10, 0, 3)]},
                },
            },
        }


class CalvinOracleObjectLocatorTest(unittest.TestCase):
    def test_classifies_table_slider_and_drawer_support(self):
        self.assertEqual(
            CalvinOracleObjectLocator(Adapter()).locate_objects(),
            {
                "red_block": "table",
                "blue_block": "slider",
                "pink_block": "drawer",
            },
        )

    def test_gripper_contact_takes_precedence_over_support(self):
        info = Adapter().get_info()
        info["scene_info"]["movable_objects"]["block_pink"]["contacts"].append(
            (0, 0, 99, 0, 0)
        )
        adapter = type("StaticAdapter", (), {"get_info": lambda self: info})()
        self.assertEqual(
            CalvinOracleObjectLocator(adapter).locate_objects()["pink_block"],
            "gripper",
        )


if __name__ == "__main__":
    unittest.main()
