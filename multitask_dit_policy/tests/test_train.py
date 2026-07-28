import unittest
from types import SimpleNamespace

from multitask_dit_policy.train import (
    _split_episode_indices,
    _validate_dataset_compatibility,
    _validate_validation_config,
)


class SplitEpisodeIndicesTest(unittest.TestCase):
    def test_is_deterministic_and_disjoint(self):
        train_indices, val_indices = _split_episode_indices(20, 0.2, seed=17)

        self.assertEqual(len(train_indices), 16)
        self.assertEqual(len(val_indices), 4)
        self.assertTrue(set(train_indices).isdisjoint(val_indices))
        self.assertEqual(sorted(train_indices + val_indices), list(range(20)))
        self.assertEqual((train_indices, val_indices), _split_episode_indices(20, 0.2, seed=17))

    def test_keeps_at_least_one_train_episode(self):
        train_indices, val_indices = _split_episode_indices(2, 0.99, seed=17)

        self.assertEqual(len(train_indices), 1)
        self.assertEqual(len(val_indices), 1)

    def test_rejects_invalid_ratio(self):
        for ratio in (-0.1, 1.0, 1.1):
            with self.subTest(ratio=ratio), self.assertRaisesRegex(ValueError, "val_ratio"):
                _split_episode_indices(10, ratio, seed=17)


class ExternalValidationConfigTest(unittest.TestCase):
    def test_external_validation_requires_zero_ratio(self):
        with self.assertRaisesRegex(ValueError, "val_ratio must be 0"):
            _validate_validation_config("/tmp/validation", 0.05)

    def test_accepts_external_validation_with_zero_ratio(self):
        _validate_validation_config("/tmp/validation", 0)

    def test_accepts_internal_validation(self):
        _validate_validation_config(None, 0.05)


class DatasetCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.features = {
            "observation.state": {"dtype": "float32", "shape": [15]},
            "action": {"dtype": "float32", "shape": [7]},
        }

    def metadata(self, fps=30, features=None):
        return SimpleNamespace(fps=fps, features=features or self.features)

    def test_accepts_matching_features_and_fps(self):
        _validate_dataset_compatibility(
            self.metadata(),
            self.metadata(),
            {"observation.state", "action"},
        )

    def test_rejects_different_fps(self):
        with self.assertRaisesRegex(ValueError, "FPS differ"):
            _validate_dataset_compatibility(
                self.metadata(fps=30),
                self.metadata(fps=15),
                {"observation.state", "action"},
            )

    def test_rejects_different_feature_shape(self):
        val_features = {
            **self.features,
            "action": {"dtype": "float32", "shape": [6]},
        }
        with self.assertRaisesRegex(ValueError, "action.shape"):
            _validate_dataset_compatibility(
                self.metadata(),
                self.metadata(features=val_features),
                {"observation.state", "action"},
            )


if __name__ == "__main__":
    unittest.main()
