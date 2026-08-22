import json
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import draccus
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

from multitask_dit_policy.model.observation_encoder import TaskIDEncoder
from multitask_dit_policy.model.model import MultiTaskDiTPolicy
from multitask_dit_policy.utils.configuration import (
    CLIPTextEncoderConfig,
    MultiTaskDiTConfig,
    ObservationEncoderConfig,
    TaskIDTextEncoderConfig,
    TransformerConfig,
)


class TaskIDEncoderTest(unittest.TestCase):
    def config(self, **overrides):
        values = {
            "num_tasks": 3,
            "id_feature_key": "calvin.task_id",
            "task_map_path": None,
            "task_to_id": {"task zero": 0, "task one": 1, "task two": 2},
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_numeric_and_string_inputs_resolve_to_same_embedding(self):
        encoder = TaskIDEncoder(self.config(), projection_dim=16)
        numeric = encoder(torch.tensor([[0], [2]]))
        strings = encoder(["task zero", "task two"])
        self.assertTrue(torch.equal(numeric, strings))

    def test_rejects_unknown_label_and_out_of_range_id(self):
        encoder = TaskIDEncoder(self.config(), projection_dim=16)
        with self.assertRaisesRegex(KeyError, "Unknown closed-set task"):
            encoder(["missing"])
        with self.assertRaisesRegex(ValueError, "must be in"):
            encoder(torch.tensor([3]))

    def test_rejects_task_change_across_observation_steps(self):
        encoder = TaskIDEncoder(self.config(), projection_dim=16)
        with self.assertRaisesRegex(ValueError, "constant"):
            encoder(torch.tensor([[[0], [1]]]))

    def test_resolves_file_mapping_into_checkpoint_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            path.write_text(json.dumps({"task_to_id": {"A": 0, "B": 1, "C": 2}}))
            config = self.config(task_map_path=str(path), task_to_id={})
            TaskIDEncoder(config, projection_dim=16)
            self.assertEqual(config.task_to_id, {"a": 0, "b": 1, "c": 2})


class TaskIDPolicyCheckpointTest(unittest.TestCase):
    def make_config(self, map_path):
        return MultiTaskDiTConfig(
            n_obs_steps=2,
            horizon=4,
            n_action_steps=2,
            observation_encoder=ObservationEncoderConfig(
                text=TaskIDTextEncoderConfig(
                    num_tasks=3,
                    task_map_path=str(map_path),
                )
            ),
            transformer=TransformerConfig(
                hidden_dim=32,
                num_layers=1,
                num_heads=4,
                dropout=0.0,
            ),
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
            },
            output_features={
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            },
            device="cpu",
        )

    def test_construct_forward_save_and_reload_without_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_path = root / "source_map.json"
            map_path.write_text(
                json.dumps({"task_to_id": {"zero": 0, "one": 1, "two": 2}})
            )
            policy = MultiTaskDiTPolicy(self.make_config(map_path))
            self.assertEqual(policy.task_conditioning_type, "task_id")
            self.assertNotIn("CLIPTextModel", repr(policy.observation_encoder.text_encoder))

            loss, _ = policy(
                {
                    OBS_STATE: torch.zeros(2, 2, 15),
                    ACTION: torch.zeros(2, 4, 7),
                    "calvin.task_id": torch.tensor([0, 2]),
                }
            )
            self.assertTrue(torch.isfinite(loss))

            checkpoint = root / "checkpoint"
            policy.save(checkpoint)
            self.assertTrue((checkpoint / "task_id_map.json").is_file())
            saved_map = json.loads((checkpoint / "task_id_map.json").read_text())
            self.assertEqual(saved_map["task_to_id"], {"one": 1, "two": 2, "zero": 0})

            # Prove reload is self-contained rather than accidentally reading
            # the original training-side mapping path.
            map_path.unlink()
            loaded = MultiTaskDiTPolicy.load(checkpoint)
            self.assertEqual(loaded.task_conditioning_type, "task_id")
            self.assertEqual(loaded.observation_encoder.text_encoder.task_to_id["two"], 2)

    def test_legacy_clip_config_still_decodes_as_clip(self):
        config = MultiTaskDiTConfig()
        stream = io.StringIO()
        with draccus.config_type("json"):
            draccus.dump(config, stream, indent=2)
        document = json.loads(stream.getvalue())
        document["observation_encoder"]["text"].pop("conditioning_type", None)
        encoded = io.StringIO(json.dumps(document))
        with draccus.config_type("json"):
            decoded = draccus.load(MultiTaskDiTConfig, encoded)
        self.assertIsInstance(decoded.observation_encoder.text, CLIPTextEncoderConfig)
        self.assertEqual(decoded.observation_encoder.text.conditioning_type, "clip")


if __name__ == "__main__":
    unittest.main()
