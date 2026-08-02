import json
from pathlib import Path
import sys
import unittest


WORKSPACE = Path(__file__).resolve().parents[2]
REMOTE_POLICY = WORKSPACE / "calvin" / "evaluation" / "remote_policy"
sys.path.insert(0, str(REMOTE_POLICY))

from task_suite import ConfiguredCalvinSuite  # noqa: E402


class SpecialistManifestTest(unittest.TestCase):
    def load(self, name):
        return json.loads(
            (WORKSPACE / "datasets" / name / "manifests/segments.json").read_text()
        )

    def test_push_rotate_has_all_tasks_and_minimum_data(self):
        manifest = self.load("calvin_push_rotate_12")
        self.assertEqual(len(manifest["tasks"]), 12)
        for split, minimum in (("training", 25), ("validation", 5)):
            counts = manifest["splits"][split]["summary"]["by_task"]
            self.assertEqual(len(counts), 12)
            self.assertTrue(all(row["retained"] >= minimum for row in counts.values()))

    def test_retained_segments_are_unique_and_oracle_consistent(self):
        for name in ("calvin_push_rotate_12", "calvin_stack_place_3"):
            manifest = self.load(name)
            for split in ("training", "validation"):
                rows = manifest["splits"][split]["segments"]
                self.assertEqual(len(rows), len({row["segment_id"] for row in rows}))
                self.assertTrue(all(row["oracle_consistent"] for row in rows))

    def test_stack_is_destination_only_and_stable(self):
        manifest = self.load("calvin_stack_place_3")
        pairs = set()
        for split in ("training", "validation"):
            for row in manifest["splits"][split]["segments"]:
                self.assertEqual(set(row["mask_roles"]), {"destination"})
                if row["calvin_task"] != "stack_block":
                    continue
                self.assertEqual(len(row["feasible_destinations"]), 2)
                self.assertIn(
                    row["demonstrated_destination"], row["feasible_destinations"]
                )
                self.assertGreaterEqual(row["destination_contact_frames"], 3)
                if split == "training":
                    pairs.add(
                        (row["source_object"], row["demonstrated_destination"])
                    )
        self.assertEqual(len(pairs), 6)

    def test_specialist_evaluation_suites_use_manifest_instructions_and_roles(self):
        push_suite = ConfiguredCalvinSuite(
            WORKSPACE / "datasets/calvin_push_rotate_12/configs/evaluation.yaml"
        )
        self.assertEqual(len(push_suite.task_names()), 12)
        push_episode = push_suite.episodes("push_red_block_left", count=1)[0]
        self.assertEqual(push_episode.instruction, "push the red block to the left")
        self.assertEqual(
            push_episode.mask_roles,
            {"target": [{"movable": "block_red"}]},
        )

        stack_suite = ConfiguredCalvinSuite(
            WORKSPACE / "datasets/calvin_stack_place_3/configs/evaluation.yaml"
        )
        self.assertEqual(len(stack_suite.task_names()), 3)
        stack_episodes = stack_suite.episodes("stack_block")
        self.assertTrue(stack_episodes)
        self.assertTrue(
            all(set(episode.mask_roles) == {"destination"} for episode in stack_episodes)
        )
        self.assertTrue(
            all(
                len(episode.mask_roles["destination"]) == 1
                for episode in stack_episodes
            )
        )


if __name__ == "__main__":
    unittest.main()
