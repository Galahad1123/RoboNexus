"""Tests for the zero-dependency offline evaluation dashboard."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from tests.paths import WORKSPACE


SPEC = importlib.util.spec_from_file_location(
    "generate_eval_report", WORKSPACE / "scripts/agent/generate_eval_report.py"
)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class OfflineEvalReportTest(unittest.TestCase):
    def make_sequence(self, root, name="sequence_001"):
        sequence = root / name
        artifact = sequence / "artifacts/session"
        artifact.mkdir(parents=True)
        (artifact / "000009_subtask_001_attempt_01.mp4").write_bytes(b"")
        (artifact / "000006_grounding_preview.png").write_bytes(b"")
        session = {
            "allowed_skill_ids": ["turn_on_led"],
            "goal_instruction": "turn on <script>alert(1)</script>",
            "session_status": "succeeded",
        }
        result = {
            "effective_status": "succeeded",
            "session_before": session,
            "session_after": session,
            "final_messages": [
                {"role": "assistant", "function_call": {"name": "execute_skill", "arguments": '{"skill_id":"turn_on_led"}'}},
                {"role": "function", "content": json.dumps({"status": "succeeded", "data": {"logical_subtask_id": "subtask_001", "attempt": 1, "policy_id": "general", "steps": 12}})},
            ],
        }
        (sequence / "qwen_agent_result.json").write_text(json.dumps(result))
        (sequence / "runtime_result.json").write_text(json.dumps({"session_status": "succeeded", "environment_steps": 12}))
        events = [
            {"event": "tool_call_started"},
            {"event": "verification_result", "episode_id": "subtask_001", "attempt": 1, "status": "success"},
            {"event": "skill_achieved", "logical_subtask_id": "subtask_001", "skill_id": "turn_on_led"},
            {"event": "goal_succeeded", "reason": "done"},
        ]
        (sequence / "events.jsonl").write_text("".join(json.dumps(item) + "\n" for item in events))
        return sequence

    def test_sequence_report_links_media_and_escapes_model_text(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = self.make_sequence(Path(directory))
            models = REPORT.generate(sequence)
            document = (sequence / "report.html").read_text()
            self.assertEqual(models[0]["status"], "succeeded")
            self.assertIn("000009_subtask_001_attempt_01.mp4", document)
            self.assertIn("policy: <code>general</code>", document)
            self.assertNotIn("<script>alert(1)</script>", document)
            self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", document)

    def test_recursive_report_builds_filterable_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_sequence(root)
            REPORT.generate(root, recursive=True)
            document = (root / "index.html").read_text()
            self.assertIn("Success rate", document)
            self.assertIn("sequence_001/report.html", document)
            self.assertIn('id="status"', document)
            self.assertIn('class="run-link"', document)

    def test_index_compacts_long_run_name_but_preserves_it_in_tooltip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_name = "sequence_0012_lift_blue_block_slider__place_in_slider__turn_on_lightbulb"
            self.make_sequence(root, full_name)
            REPORT.generate(root, recursive=True)
            document = (root / "index.html").read_text()
            self.assertIn(">sequence_0012</a>", document)
            self.assertIn('title="{}"'.format(full_name), document)
            self.assertIn("table-layout:fixed", document)

    def test_multimodal_function_result_extracts_json_text(self):
        result = REPORT.normalize_tool_result(
            [{"text": '{"status":"succeeded"}'}, {"image": "/tmp/view.png"}]
        )
        self.assertEqual(result["status"], "succeeded")

    def test_failed_execution_is_not_rendered_unknown(self):
        self.assertEqual(
            REPORT._execution_state(
                {"success": False, "verification": {"status": "failure"}}, None
            ),
            "failed",
        )

    def test_stage_uses_actual_skill_not_logical_subtask_number(self):
        skills = ["open_drawer", "turn_on_led", "move_slider_right"]
        self.assertEqual(REPORT._skill_stage("turn_on_led", skills), "stage 2/3")
        self.assertEqual(REPORT._skill_stage("lift_red_block_table", skills), "recovery skill")


if __name__ == "__main__":
    unittest.main()
