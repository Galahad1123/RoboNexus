import unittest

from adapters.qwen_agent.remote_tool import (
    _normalize_tool_params,
    _qwen_compatible_schema,
    _result_content,
)
from adapters.qwen_agent.single_skill_runner import _continuation_content


class RemoteToolMultimodalResultTest(unittest.TestCase):
    def test_continuation_compacts_history_and_keeps_latest_images(self):
        transcript = [
            {
                "role": "function",
                "name": "inspect_scene",
                "content": [
                    {"text": '{"status":"succeeded","data":{"observation":{"observation_id":"obs_9"}}}'},
                    {"text": "CURRENT_STATIC_VIEW"},
                    {"image": "/tmp/static.png"},
                    {"text": "CURRENT_WRIST_VIEW"},
                    {"image": "/tmp/wrist.png"},
                ],
            },
            {
                "role": "function",
                "name": "execute_skill",
                "content": '{"status":"failed","data":{"skill_id":"open_drawer","attempt":3,"termination_reason":"timeout"}}',
            },
        ]
        content = _continuation_content(
            {"goal_id": "g"},
            {"session_status": "active", "latest_observation_id": "obs_9", "budget": {}},
            transcript,
        )
        dumped = [item.model_dump() for item in content]
        self.assertEqual([item.get("image") for item in dumped if item.get("image")], ["/tmp/static.png", "/tmp/wrist.png"])
        self.assertIn('"attempt": 3', dumped[0]["text"])
    def test_grounding_description_is_optional_for_qwen_and_normalized(self):
        schema = {
            "type": "object",
            "properties": {
                "proposals": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "role",
                            "representation",
                            "semantic_id",
                            "description",
                        ],
                    },
                }
            },
            "required": ["proposals"],
        }
        adapted = _qwen_compatible_schema(schema)
        self.assertNotIn(
            "description",
            adapted["properties"]["proposals"]["items"]["required"],
        )
        normalized = _normalize_tool_params(
            "ground_targets",
            {
                "proposals": [
                    {
                        "role": "target",
                        "representation": "semantic",
                        "semantic_id": "led_button",
                    }
                ]
            },
        )
        self.assertEqual(
            normalized["proposals"][0]["description"],
            "Agent-selected target led_button",
        )

    def test_inspect_scene_injects_fresh_static_and_wrist_images(self):
        result = {
            "status": "succeeded",
            "data": {"observation": {"observation_id": "obs_000002"}},
            "artifacts": [
                {"kind": "current_static", "uri": "artifact://x/static.png"},
                {"kind": "current_wrist", "uri": "artifact://x/wrist.png"},
            ],
            "artifact_paths": ["/tmp/static.png", "/tmp/wrist.png"],
        }
        content = _result_content("inspect_scene", result)
        self.assertEqual(len(content), 5)
        self.assertIn('"observation_id": "obs_000002"', content[0].text)
        self.assertEqual(content[2].image, "/tmp/static.png")
        self.assertEqual(content[4].image, "/tmp/wrist.png")

    def test_non_visual_tools_keep_compact_json_result(self):
        content = _result_content(
            "execute_skill", {"status": "succeeded", "data": {"success": True}}
        )
        self.assertIsInstance(content, str)
        self.assertIn('"success": true', content)

    def test_failed_inspection_does_not_inject_stale_images(self):
        content = _result_content(
            "inspect_scene",
            {
                "status": "failed",
                "artifacts": [{"kind": "current_static"}],
                "artifact_paths": ["/tmp/stale.png"],
            },
        )
        self.assertIsInstance(content, str)


if __name__ == "__main__":
    unittest.main()
