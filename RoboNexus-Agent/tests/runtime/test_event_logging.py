"""Tests for active structured event loggers."""

import json
import tempfile
import unittest
from pathlib import Path

from core.logging import InMemoryEventLogger, JsonlEventLogger


class EventLoggingTest(unittest.TestCase):
    def test_jsonl_logger_writes_structured_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logger = JsonlEventLogger(path, "test-run")
            logger.emit(
                "attempt_failed",
                episode_id="validation-1",
                attempt=1,
                reason="timeout",
            )
            logger.close()
            event = json.loads(path.read_text().strip())
            self.assertEqual(event["run_id"], "test-run")
            self.assertEqual(event["event"], "attempt_failed")
            self.assertEqual(event["attempt"], 1)
            self.assertIn("timestamp", event)

    def test_memory_logger_is_suitable_for_runtime_tests(self):
        logger = InMemoryEventLogger()
        logger.emit("recovery_started", attempt=2)
        self.assertEqual(logger.events[0]["event"], "recovery_started")


if __name__ == "__main__":
    unittest.main()
