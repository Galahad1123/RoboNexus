"""Tests for crash-safe result writes and interrupted-run recovery."""

import json
import tempfile
import unittest
from pathlib import Path

from core.logging import atomic_write_json, recover_agent_result


class ResultStoreTest(unittest.TestCase):
    def test_atomic_write_replaces_complete_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            atomic_write_json(path, {"generation": 1})
            atomic_write_json(path, {"generation": 2, "complete": True})
            self.assertEqual(json.loads(path.read_text()), {"generation": 2, "complete": True})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_runtime_success_recovers_interrupted_agent_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            atomic_write_json(
                output / "qwen_agent_checkpoint.json",
                {"run_state": "running", "num_agent_updates": 7},
            )
            atomic_write_json(
                output / "runtime_result.json",
                {"session_status": "succeeded", "termination_reason": "goal_achieved"},
            )
            (output / "events.jsonl").write_text('{"event":"goal_succeeded"}\n')

            report = recover_agent_result(output)

            self.assertEqual(report["effective_status"], "succeeded")
            self.assertEqual(report["run_state"], "completed")
            self.assertTrue(report["recovered"])
            self.assertTrue(report["recovery_sources"]["events"])
            self.assertTrue((output / "qwen_agent_result.json").is_file())

    def test_missing_artifacts_cannot_be_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                recover_agent_result(directory)

    def test_terminal_event_recovers_without_runtime_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            atomic_write_json(
                output / "qwen_agent_checkpoint.json", {"run_state": "running"}
            )
            (output / "events.jsonl").write_text(
                '{"event":"goal_succeeded"}\n{"event":', encoding="utf-8"
            )

            report = recover_agent_result(output)

            self.assertEqual(report["effective_status"], "succeeded")
            self.assertEqual(report["run_state"], "completed")


if __name__ == "__main__":
    unittest.main()
