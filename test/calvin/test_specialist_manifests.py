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

    def test_refined_rotate_requires_release_and_stable_suffix(self):
        manifest = self.load("calvin_rotate_6_release")
        for split in ("training", "validation"):
            rows = manifest["splits"][split]["segments"]
            self.assertEqual(len({row["calvin_task"] for row in rows}), 6)
            self.assertEqual(len(rows), len({row["segment_id"] for row in rows}))
            for row in rows:
                self.assertTrue(row["oracle_consistent"])
                self.assertIsNotNone(row["oracle_success_frame"])
                self.assertIsNotNone(row["release_frame"])
                self.assertGreaterEqual(row["stable_suffix_frames"], 15)
                self.assertLessEqual(row["stable_suffix_frames"], 30)
                self.assertLessEqual(row["release_frame"], row["original_end"] + 15)

    def test_mixed_stack_has_both_variants_and_dynamic_source_role(self):
        manifest = self.load("calvin_stack_pick_place")
        training = manifest["splits"]["training"]["segments"]
        variants = {row["episode_variant"] for row in training}
        self.assertEqual(
            variants,
            {
                "table_pick",
                "full_pick_and_stack",
                "held_start_stack",
                "place_in_drawer",
                "place_in_slider",
            },
        )
        variant_counts = {}
        for row in training:
            variant = row["episode_variant"]
            variant_counts[variant] = variant_counts.get(variant, 0) + 1
        self.assertEqual(variant_counts["table_pick"], 100)
        self.assertEqual(variant_counts["place_in_drawer"], 100)
        self.assertEqual(variant_counts["place_in_slider"], 100)
        pick_colors = {}
        for row in training:
            if row["episode_variant"] != "table_pick":
                continue
            task = row["calvin_task"]
            pick_colors[task] = pick_colors.get(task, 0) + 1
        self.assertEqual(sorted(pick_colors.values()), [33, 33, 34])
        pairs = set()
        for row in training:
            if row["episode_variant"] == "table_pick":
                self.assertEqual(set(row["mask_roles"]), {"source"})
                self.assertEqual(row["mask_roles"]["source"][0]["when"], "until_held")
            if row["calvin_task"] == "stack_block":
                self.assertEqual(set(row["mask_roles"]), {"source", "destination"})
                self.assertEqual(row["mask_roles"]["source"][0]["when"], "until_held")
                self.assertNotEqual(row["source_object"], row["demonstrated_destination"])
                pairs.add((row["source_object"], row["demonstrated_destination"]))
        self.assertEqual(len(pairs), 6)

    def test_refined_evaluation_suites_cover_tasks_and_stack_variants(self):
        rotate_suite = ConfiguredCalvinSuite(
            WORKSPACE / "datasets/calvin_rotate_6_release/configs/evaluation.yaml"
        )
        self.assertEqual(len(rotate_suite.task_names()), 6)
        self.assertTrue(
            all(
                episode.episode_variant == "rotate_release_complete"
                for task in rotate_suite.task_names()
                for episode in rotate_suite.episodes(task)
            )
        )
        self.assertTrue(
            all(
                episode.success_guard["gripper"] == "open"
                and episode.success_guard["consecutive_steps"] == 3
                for task in rotate_suite.task_names()
                for episode in rotate_suite.episodes(task)
            )
        )

        mixed_suite = ConfiguredCalvinSuite(
            WORKSPACE / "datasets/calvin_stack_pick_place/configs/evaluation.yaml"
        )
        self.assertEqual(len(mixed_suite.task_names()), 6)
        stack_episodes = mixed_suite.episodes("stack_block")
        self.assertEqual(
            {episode.episode_variant for episode in stack_episodes},
            {"full_pick_and_stack", "held_start_stack"},
        )
        self.assertEqual(
            len(
                mixed_suite.episodes(
                    "stack_block",
                    episode_variants=["full_pick_and_stack"],
                )
            ),
            4,
        )
        self.assertTrue(
            all(
                episode.mask_roles["source"][0]["when"] == "until_held"
                for episode in stack_episodes
            )
        )

    def test_third_generation_merges_rotate_language_and_keeps_source_visible(self):
        rotate = self.load("calvin_rotate_2_release")
        self.assertEqual(
            {item["instruction"] for item in rotate["tasks"]},
            {
                "rotate the black-outlined block counterclockwise",
                "rotate the black-outlined block clockwise",
            },
        )
        for split in ("training", "validation"):
            self.assertEqual(
                len(rotate["splits"][split]["segments"]),
                len(
                    self.load("calvin_rotate_6_release")["splits"][split]["segments"]
                ),
            )

        persistent = self.load("calvin_stack_pick_place_persistent")
        for split in ("training", "validation"):
            for row in persistent["splits"][split]["segments"]:
                if row["episode_variant"] in {
                    "table_pick",
                    "full_pick_and_stack",
                    "held_start_stack",
                }:
                    self.assertEqual(
                        row["mask_roles"]["source"],
                        [{"movable": row["source_object"]}],
                    )

    def test_third_generation_evaluation_suites(self):
        rotate_suite = ConfiguredCalvinSuite(
            WORKSPACE / "datasets/calvin_rotate_2_release/configs/evaluation.yaml"
        )
        self.assertEqual(len(rotate_suite.task_names()), 6)
        rotate_episodes = [
            episode
            for task in rotate_suite.task_names()
            for episode in rotate_suite.episodes(task)
        ]
        self.assertEqual(
            {episode.instruction for episode in rotate_episodes},
            {
                "rotate the black-outlined block counterclockwise",
                "rotate the black-outlined block clockwise",
            },
        )
        self.assertTrue(
            all(
                episode.success_guard["gripper"] == "open"
                and episode.success_guard["consecutive_steps"] == 3
                for episode in rotate_episodes
            )
        )

        persistent_suite = ConfiguredCalvinSuite(
            WORKSPACE
            / "datasets/calvin_stack_pick_place_persistent/configs/evaluation.yaml"
        )
        self.assertEqual(len(persistent_suite.task_names()), 6)
        stack_episodes = persistent_suite.episodes("stack_block")
        self.assertEqual(
            {episode.episode_variant for episode in stack_episodes},
            {"full_pick_and_stack", "held_start_stack"},
        )
        self.assertTrue(
            all(
                episode.mask_roles["source"]
                and "when" not in episode.mask_roles["source"][0]
                for episode in stack_episodes
            )
        )

    def test_d34_routing_and_new_manifests(self):
        inventory = json.loads(
            (WORKSPACE / "datasets/calvin_d34/reports/task_inventory.json").read_text()
        )
        self.assertEqual(inventory["num_tasks"], 34)
        self.assertTrue(inventory["routing_is_disjoint"])
        routed = [task for tasks in inventory["routing"].values() for task in tasks]
        self.assertEqual(len(routed), len(set(routed)))
        self.assertEqual(
            {group: len(tasks) for group, tasks in inventory["routing"].items()},
            {"push": 6, "rotate": 6, "stack": 1, "general": 21},
        )

        push = self.load("calvin_push_6")
        self.assertEqual(len(push["tasks"]), 6)
        self.assertEqual(
            {item["instruction"] for item in push["tasks"]},
            {
                "push the blue block to the left",
                "push the blue block to the right",
                "push the pink block to the left",
                "push the pink block to the right",
                "push the red block to the left",
                "push the red block to the right",
            },
        )
        general = self.load("calvin_general_21")
        self.assertEqual(len(general["tasks"]), 21)
        for manifest in (push, general):
            for split in ("training", "validation"):
                rows = manifest["splits"][split]["segments"]
                self.assertEqual(len(rows), len({row["segment_id"] for row in rows}))
                self.assertTrue(all(row["oracle_consistent"] for row in rows))
                self.assertTrue(all(row["mask_roles"] for row in rows))

    def test_d34_evaluation_suites_and_extended_stack(self):
        push_suite = ConfiguredCalvinSuite(
            WORKSPACE / "datasets/calvin_push_6/configs/evaluation.yaml"
        )
        general_suite = ConfiguredCalvinSuite(
            WORKSPACE / "datasets/calvin_general_21/configs/evaluation.yaml"
        )
        extended_suite = ConfiguredCalvinSuite(
            WORKSPACE
            / "datasets/calvin_full_stack_extended_eval/configs/evaluation.yaml"
        )
        self.assertEqual(len(push_suite.task_names()), 6)
        self.assertEqual(len(general_suite.task_names()), 21)
        extended = extended_suite.episodes("stack_block")
        self.assertEqual(len(extended), 21)
        self.assertTrue(
            all(row.episode_variant == "full_pick_and_stack_extended" for row in extended)
        )
        manifest = json.loads(
            (
                WORKSPACE
                / "datasets/calvin_full_stack_extended_eval/manifests/segments.json"
            ).read_text()
        )
        rows = manifest["splits"]["validation"]["segments"]
        self.assertTrue(manifest["evaluation_only"])
        self.assertTrue(all(not row["training_eligible"] for row in rows))
        self.assertEqual(
            {row["evaluation_tier"] for row in rows},
            {"official", "extended_backtracked"},
        )

    def test_abc_manipulation_d_suite_uses_physical_validation_split(self):
        suite = ConfiguredCalvinSuite(
            WORKSPACE
            / "datasets/calvin_abc_manipulation_13_clean/configs/evaluation_d.yaml"
        )
        self.assertEqual(suite.split, "d_validation")
        self.assertEqual(suite.source_split, "validation")
        self.assertEqual(len(suite.task_names()), 13)
        self.assertEqual(len(suite.episodes("stack_block")), 2)
        episodes = [
            episode
            for task in suite.task_names()
            for episode in suite.episodes(task)
        ]
        self.assertEqual(len(episodes), 44)
        self.assertTrue(all(episode.source_frame.is_file() for episode in episodes))
        self.assertTrue(
            all("task_ABC_D/validation" in str(episode.source_frame) for episode in episodes)
        )

    def test_abc_general_d_suite_uses_clean_d_manifest(self):
        suite = ConfiguredCalvinSuite(
            WORKSPACE
            / "datasets/calvin_abc_general_21_clean/configs/evaluation_d.yaml"
        )
        self.assertEqual(suite.split, "d_validation")
        self.assertEqual(suite.source_split, "validation")
        self.assertEqual(len(suite.task_names()), 21)
        episodes = [
            episode
            for task in suite.task_names()
            for episode in suite.episodes(task)
        ]
        self.assertEqual(len(episodes), 47)
        self.assertTrue(all(episode.source_frame.is_file() for episode in episodes))
        self.assertTrue(
            all("task_ABC_D/validation" in str(episode.source_frame) for episode in episodes)
        )
        place = suite.episodes("place_in_drawer", count=1)[0]
        self.assertEqual(place.instruction, "place the held block in the drawer")
        self.assertEqual(place.success_guard["consecutive_steps"], 3)


if __name__ == "__main__":
    unittest.main()
