"""Completeness and validation tests for CALVIN-D34 configuration."""

import json
import tempfile
import unittest
from pathlib import Path

from core.config import ConfigValidationError
from core.policies import load_policy_catalog
from core.skills import CALVIN_D34_ROUTING, CALVIN_D34_SKILL_IDS, load_skill_registry
from tests.paths import AGENT_ROOT, WORKSPACE


SKILLS_CONFIG = AGENT_ROOT / "configs/calvin/skills_d34.yaml"
POLICIES_CONFIG = AGENT_ROOT / "configs/calvin/policies.yaml"
INVENTORY = WORKSPACE / "datasets/calvin_d34/reports/task_inventory.json"


class CalvinD34ConfigTest(unittest.TestCase):
    def test_registry_exactly_matches_audited_inventory_and_routing(self):
        registry = load_skill_registry(SKILLS_CONFIG)
        inventory = json.loads(INVENTORY.read_text())
        inventory_ids = {
            skill_id
            for skill_ids in inventory["routing"].values()
            for skill_id in skill_ids
        }
        self.assertEqual(len(registry), 34)
        self.assertEqual(set(registry.skill_ids()), CALVIN_D34_SKILL_IDS)
        self.assertEqual(set(registry.skill_ids()), inventory_ids)
        self.assertEqual(registry.policy_groups(), ("general", "push", "rotate", "stack"))
        self.assertEqual(
            set(registry.categories()),
            {"articulation", "lift", "place", "push", "rotate", "stack", "switch"},
        )
        for policy_group, skill_ids in CALVIN_D34_ROUTING.items():
            self.assertEqual(
                {skill.skill_id for skill in registry.by_policy_group(policy_group)},
                set(skill_ids),
            )
            self.assertEqual(set(skill_ids), set(inventory["routing"][policy_group]))

    def test_registry_applies_defaults_and_preserves_overrides_and_guards(self):
        registry = load_skill_registry(SKILLS_CONFIG)
        general = registry.get("open_drawer")
        push = registry.get("push_red_block_left")
        rotate = registry.get("rotate_pink_block_right")
        self.assertEqual(general.max_steps, 150)
        self.assertEqual(push.max_steps, 120)
        self.assertEqual(general.max_attempts, 3)
        self.assertEqual(general.replan_interval, 16)
        self.assertIsNone(general.success_guard)
        self.assertEqual(rotate.success_guard["gripper"], "open")
        self.assertEqual(rotate.success_guard["consecutive_steps"], 3)
        self.assertEqual(general.grounding_roles, ("target",))
        self.assertEqual(registry.get("stack_block").grounding_roles, ("source", "destination"))
        self.assertEqual(
            registry.get("lift_blue_block_drawer").instruction,
            "pick up the black-outlined block from the drawer",
        )
        self.assertEqual(
            registry.get("lift_blue_block_drawer").planner_instruction,
            "pick up the blue block from the drawer",
        )
        self.assertEqual(
            registry.get("open_drawer").planner_instruction,
            registry.get("open_drawer").instruction,
        )

    def test_registry_preserves_structured_planner_guidance(self):
        registry = load_skill_registry(SKILLS_CONFIG)
        slider = registry.get("place_in_slider").planner_guidance
        self.assertIsNotNone(slider)
        self.assertIn("destination grounding only", slider.grounding[0])
        self.assertIn("slider_interior", slider.grounding[1])
        self.assertIn("holding a block", slider.planning[0])
        drawer = registry.get("place_in_drawer").planner_guidance
        self.assertIn("role=destination", drawer.grounding[0])
        self.assertIn("drawer_interior", drawer.grounding[1])
        self.assertEqual(
            registry.get("turn_on_led").planner_guidance,
            registry.get("turn_off_led").planner_guidance,
        )
        self.assertIn(
            "light_switch",
            registry.get("turn_on_lightbulb").planner_guidance.grounding[0],
        )
        slider_lift = registry.get("lift_red_block_slider").planner_guidance
        self.assertIn("colored block", slider_lift.grounding[0])
        self.assertIn("Never ground slider_interior", slider_lift.grounding[0])

    def test_registry_preserves_conditional_recovery_catalog(self):
        registry = load_skill_registry(SKILLS_CONFIG)
        rule = registry.get("place_in_drawer").recovery[0]
        self.assertEqual(rule.strategy, "reacquire_object")
        self.assertEqual(rule.candidate_categories, ("lift",))
        self.assertEqual(rule.return_to_skill, "place_in_drawer")
        self.assertIn("open_drawer", registry.get("place_in_drawer").recovery[1].candidate_skills)

    def test_malformed_planner_guidance_is_rejected(self):
        text = SKILLS_CONFIG.read_text().replace(
            "    planner_guidance: &led_guidance\n",
            "    planner_guidance: &led_guidance\n      backend_validator: true\n",
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skills.yaml"
            path.write_text(text)
            with self.assertRaisesRegex(
                ConfigValidationError, "planner_guidance has unknown fields"
            ):
                load_skill_registry(path)

    def test_policy_catalog_covers_every_group_with_unique_endpoints(self):
        registry = load_skill_registry(SKILLS_CONFIG)
        catalog = load_policy_catalog(
            POLICIES_CONFIG,
            required_policy_ids=registry.policy_groups(),
        )
        self.assertEqual(catalog.policy_ids(), registry.policy_groups())
        self.assertEqual(
            {catalog.get(item).port for item in catalog.policy_ids()},
            {10093, 10094, 10095, 10096},
        )
        expected_checkpoints = {
            "general": "outputs/calvin_general_21_dino/checkpoint_80000",
            "push": "outputs/calvin_push_rotate_12_dino_outline/checkpoint_80000",
            "rotate": "outputs/calvin_rotate_2_release_dino_outline/checkpoint_80000",
            "stack": "outputs/calvin_stack_pick_place_persistent_outline/checkpoint_80000",
        }
        for policy_id, checkpoint in expected_checkpoints.items():
            policy = catalog.get(policy_id)
            self.assertTrue(policy.enabled)
            self.assertEqual(policy.checkpoint, checkpoint)

    def test_missing_canonical_skill_is_rejected(self):
        text = SKILLS_CONFIG.read_text()
        marker = "  stack_block:\n"
        truncated = text[: text.index(marker)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skills.yaml"
            path.write_text(truncated)
            with self.assertRaisesRegex(
                ConfigValidationError, "missing canonical IDs.*stack_block"
            ):
                load_skill_registry(path)

    def test_duplicate_yaml_key_is_rejected_before_schema_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policies.yaml"
            path.write_text(
                "version: 1\n"
                "name: duplicate\n"
                "checkpoint_base: multitask_dit_policy\n"
                "policies:\n"
                "  push: {enabled: false, checkpoint: null, host: 127.0.0.1, port: 10094}\n"
                "  push: {enabled: false, checkpoint: null, host: 127.0.0.1, port: 10095}\n"
            )
            with self.assertRaisesRegex(ConfigValidationError, "duplicate key 'push'"):
                load_policy_catalog(path)

    def test_enabled_policy_requires_checkpoint(self):
        text = POLICIES_CONFIG.read_text().replace(
            "    checkpoint: outputs/calvin_general_21_dino/checkpoint_80000",
            "    checkpoint: null",
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policies.yaml"
            path.write_text(text)
            with self.assertRaisesRegex(
                ConfigValidationError,
                "policies.general.checkpoint is required",
            ):
                load_policy_catalog(
                    path,
                    required_policy_ids=("general", "push", "rotate", "stack"),
                )

    def test_canonical_skill_cannot_be_silently_routed_to_wrong_model(self):
        text = SKILLS_CONFIG.read_text().replace(
            "  push_red_block_left:\n"
            "    category: push\n"
            "    instruction: push the red block to the left\n"
            "    oracle_task: push_red_block_left\n"
            "    policy_group: push",
            "  push_red_block_left:\n"
            "    category: push\n"
            "    instruction: push the red block to the left\n"
            "    oracle_task: push_red_block_left\n"
            "    policy_group: general",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skills.yaml"
            path.write_text(text)
            with self.assertRaisesRegex(
                ConfigValidationError,
                "skills.push_red_block_left.policy_group must be 'push'",
            ):
                load_skill_registry(path)

    def test_selected_checkpoints_pass_runtime_validation(self):
        catalog = load_policy_catalog(
            POLICIES_CONFIG,
            required_policy_ids=("general", "push", "rotate", "stack"),
        )
        catalog.validate_runtime(WORKSPACE)


if __name__ == "__main__":
    unittest.main()
