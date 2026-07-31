import json
import tempfile
import unittest
from pathlib import Path

from core.execution.retry_controller import RetryPolicy, TaskRetryController
from core.execution.schemas import (
    AttemptResult,
    MotionResult,
    MotionStatus,
    TargetRole,
    TaskSpec,
    VerificationStatus,
)
from core.logging import InMemoryEventLogger, JsonlEventLogger
from core.schemas import ObjectRole


class ExecutionContractsTest(unittest.TestCase):
    def test_interaction_role_is_available_across_contracts(self):
        self.assertEqual(ObjectRole.INTERACTION.value, "interaction")
        self.assertEqual(TargetRole.INTERACTION.value, "interaction")
        self.assertEqual(VerificationStatus.UNKNOWN.value, "unknown")

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

    def test_memory_logger_is_suitable_for_orchestrator_tests(self):
        logger = InMemoryEventLogger()
        logger.emit("recovery_started", attempt=2)
        self.assertEqual(logger.events[0]["event"], "recovery_started")

    def test_retry_controller_recovers_without_resetting_backend_state(self):
        logger = InMemoryEventLogger()
        state = {"value": 0}

        def execute(attempt):
            state["value"] += 1
            return AttemptResult(
                success=attempt == 2,
                steps=1,
                termination_reason="success" if attempt == 2 else "timeout",
                observation=state,
            )

        def recover(attempt, observation):
            self.assertIs(observation, state)
            state["value"] += 10
            return MotionResult(MotionStatus.REACHED, 1, "reached")

        attempts = TaskRetryController(logger).run(
            TaskSpec("lift", "lift the block"),
            RetryPolicy(max_attempts=3),
            execute,
            recover,
        )
        self.assertEqual(len(attempts), 2)
        self.assertTrue(attempts[-1].success)
        self.assertEqual(state["value"], 12)


if __name__ == "__main__":
    unittest.main()
