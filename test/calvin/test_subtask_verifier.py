#!/usr/bin/env python3
"""Unit tests for the isolated CALVIN subtask verifier."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from subtask_verifier import FramePair, parse_json_object, run_verification
from run_verifier_benchmark import render_html, summarize


CRITERIA = [
    {"criterion_id": "inside", "description": "The red block is inside the drawer.", "required": True},
    {"criterion_id": "released", "description": "The gripper is no longer holding the red block.", "required": True},
]


def result(status, states):
    return {
        "subtask_id": "put_block_in_drawer",
        "observation": {
            "current_frame_id": 10,
            "previous_frame_ids": [2, 6],
            "robot_motion_state": "stopped",
            "view_quality": "sufficient",
            "occlusion": "none",
            "confidence": 0.8,
        },
        "success_criteria": [
            {
                "criterion_id": criterion["criterion_id"],
                "status": state,
                "confidence": 0.8,
                "observability": "observable" if state != "unknown" else "occluded",
                "evidence_source": "both" if state != "unknown" else "none",
                "evidence": "scripted test evidence",
            }
            for criterion, state in zip(CRITERIA, states)
        ],
        "temporal_change": {
            "progress": "positive",
            "still_changing": False,
            "unexpected_change": False,
            "confidence": 0.8,
            "evidence": "scripted temporal evidence",
        },
        "decision": {
            "status": status,
            "confidence": 0.8,
            "reason": "scripted",
            "recommended_action": "terminate" if status == "success" else "observe_again",
        },
    }


class ScriptedBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.frame_counts = []
        self.goal_reference_flags = []

    def verify(
        self,
        subtask,
        success_criteria,
        failure_conditions,
        frames,
        goal_reference=None,
        goal_reference_wrist=None,
        initial_state=None,
        observation_context=None,
        include_wrist=True,
    ):
        self.frame_counts.append(len(frames))
        self.goal_reference_flags.append(goal_reference is not None)
        self.goal_reference_wrist = goal_reference_wrist
        self.initial_state = initial_state
        return self.responses.pop(0)


class SubtaskVerifierTest(unittest.TestCase):
    def setUp(self):
        image = Image.new("RGB", (8, 8), "white")
        self.current = FramePair(image, image, "current")
        self.history = [FramePair(image, image, "history_1"), FramePair(image, image, "history_2")]

    def test_success_stops_after_current_pair_without_history(self):
        backend = ScriptedBackend([result("success", ["pass", "pass"])])
        report = run_verification(backend, "put block in drawer", CRITERIA, self.current)
        self.assertEqual(report["result"]["decision"]["status"], "success")
        self.assertFalse(report["used_temporal_recheck"])
        self.assertEqual(backend.frame_counts, [1])
        self.assertGreaterEqual(report["total_latency_seconds"], 0.0)

    def test_uncertain_triggers_sparse_temporal_recheck(self):
        backend = ScriptedBackend(
            [result("success", ["pass", "pass"])]
        )
        report = run_verification(backend, "put block in drawer", CRITERIA, self.current, history=self.history)
        self.assertEqual(report["result"]["decision"]["status"], "success")
        self.assertTrue(report["used_temporal_recheck"])
        self.assertEqual(backend.frame_counts, [3])
        self.assertEqual(len(report["passes"]), 1)

    def test_goal_reference_is_forwarded_to_backend(self):
        backend = ScriptedBackend([result("success", ["pass", "pass"])])
        report = run_verification(
            backend,
            "put block in drawer",
            CRITERIA,
            self.current,
            goal_reference=self.current.static,
        )
        self.assertEqual(report["result"]["decision"]["status"], "success")
        self.assertEqual(backend.goal_reference_flags, [True])

    def test_wrist_goal_reference_is_forwarded_to_backend(self):
        backend = ScriptedBackend([result("success", ["pass", "pass"])])
        run_verification(
            backend,
            "lift red block",
            CRITERIA,
            self.current,
            goal_reference=self.current.static,
            goal_reference_wrist=self.current.wrist,
        )
        self.assertIs(backend.goal_reference_wrist, self.current.wrist)

    def test_initial_state_is_forwarded_to_both_passes(self):
        backend = ScriptedBackend(
            [result("success", ["pass", "pass"])]
        )
        run_verification(
            backend,
            "put block in drawer",
            CRITERIA,
            self.current,
            history=self.history,
            initial_state=self.current,
        )
        self.assertIs(backend.initial_state, self.current)

    def test_candidate_success_is_temporally_confirmed(self):
        backend = ScriptedBackend(
            [result("success", ["pass", "pass"])]
        )
        report = run_verification(
            backend, "put block in drawer", CRITERIA, self.current, history=self.history
        )
        self.assertTrue(report["used_temporal_recheck"])
        self.assertEqual(backend.frame_counts, [3])

    def test_in_progress_is_temporally_rechecked_by_default(self):
        backend = ScriptedBackend(
            [result("in_progress", ["fail", "fail"])]
        )
        report = run_verification(
            backend, "put block in drawer", CRITERIA, self.current, history=self.history
        )
        self.assertTrue(report["used_temporal_recheck"])
        self.assertEqual(backend.frame_counts, [3])

    def test_inconsistent_success_is_downgraded(self):
        backend = ScriptedBackend([result("success", ["pass", "unknown"])])
        report = run_verification(backend, "put block in drawer", CRITERIA, self.current)
        self.assertEqual(report["result"]["decision"]["status"], "need_more_evidence")
        self.assertEqual(report["result"]["decision"]["recommended_action"], "observe_again")

    def test_pass_with_contradictory_evidence_is_forced_to_fail(self):
        criteria = [
            {
                "criterion_id": "opening_closed",
                "description": "The drawer is closed.",
                "required": True,
                "fail_if_evidence_contains": ["drawer is open", "not closed"],
            }
        ]
        response = result("success", ["pass"])
        response["success_criteria"][0]["criterion_id"] = "opening_closed"
        response["success_criteria"][0]["evidence"] = "The drawer is open, not closed."
        backend = ScriptedBackend([response])
        report = run_verification(backend, "close drawer", criteria, self.current)
        criterion = report["result"]["success_criteria"][0]
        self.assertEqual(criterion["status"], "fail")
        self.assertIn("contract_correction", criterion)
        self.assertEqual(report["result"]["decision"]["status"], "in_progress")

    def test_common_schema_aliases_are_normalized(self):
        response = result("success", ["pass", "pass"])
        response["success_criteria"][0]["observability"] = "fully observable"
        response["success_criteria"][0]["status"] = "satisfied"
        response["success_criteria"][0]["evidence_source"] = "static view"
        backend = ScriptedBackend([response])
        report = run_verification(backend, "put block in drawer", CRITERIA, self.current)
        criterion = report["result"]["success_criteria"][0]
        self.assertEqual(criterion["observability"], "observable")
        self.assertEqual(criterion["status"], "pass")
        self.assertEqual(criterion["evidence_source"], "static")

    def test_unknown_observability_is_safely_downgraded(self):
        response = result("success", ["pass", "pass"])
        response["success_criteria"][0]["observability"] = "mostly_guessable"
        backend = ScriptedBackend([response])
        report = run_verification(backend, "put block in drawer", CRITERIA, self.current)
        criterion = report["result"]["success_criteria"][0]
        self.assertEqual(criterion["observability"], "partially_observable")
        self.assertEqual(criterion["status"], "unknown")
        self.assertEqual(report["result"]["decision"]["status"], "need_more_evidence")

    def test_json_code_fence_is_accepted(self):
        parsed = parse_json_object('```json\n{"status": "success"}\n```')
        self.assertEqual(parsed["status"], "success")

    def test_json_with_trailing_prose_is_accepted(self):
        parsed = parse_json_object('{"status": "success"}\nDone.')
        self.assertEqual(parsed["status"], "success")

    def test_static_html_embeds_images_and_verdict(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "frame.png"
            Image.new("RGB", (8, 8), "red").save(image_path)
            verification = {
                "result": result("success", ["pass", "pass"]),
                "total_latency_seconds": 1.25,
                "used_temporal_recheck": False,
                "passes": [],
            }
            row = {
                "case_id": "demo__end_positive",
                "instruction": "put block in drawer",
                "static_image": "frame.png",
                "wrist_image": "frame.png",
                "initial_static_image": "frame.png",
                "initial_wrist_image": "frame.png",
                "history": [],
                "oracle_success": True,
                "qwen_success": True,
                "agreement": True,
                "verification": verification,
            }
            report = {"rows": [row], "summary": summarize([row])}
            destination = root / "report.html"
            render_html(report, root / "manifest.json", destination)
            document = destination.read_text(encoding="utf-8")
            self.assertIn("data:image/png;base64,", document)
            self.assertIn("Oracle: <b>success</b>", document)
            self.assertIn("Qwen: <b>success</b>", document)


if __name__ == "__main__":
    unittest.main()
