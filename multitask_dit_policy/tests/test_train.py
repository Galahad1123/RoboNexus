import unittest
from types import SimpleNamespace

from multitask_dit_policy.train import (
    _build_task_balanced_sampler,
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


class TaskBalancedSamplerTest(unittest.TestCase):
    def test_balances_tasks_instead_of_frame_counts(self):
        episodes = [
            {
                "tasks": ["frequent"],
                "dataset_from_index": 0,
                "dataset_to_index": 900,
            },
            {
                "tasks": ["rare"],
                "dataset_from_index": 900,
                "dataset_to_index": 1000,
            },
        ]
        sampler = _build_task_balanced_sampler(
            episodes, [0, 1], drop_n_last_frames=0, seed=42
        )
        sampled = list(sampler)
        frequent = sum(index < 900 for index in sampled)
        self.assertEqual(len(sampled), 1000)
        self.assertLess(abs(frequent / len(sampled) - 0.5), 0.06)

    def test_is_deterministic_for_a_fresh_sampler(self):
        episodes = [
            {"tasks": ["a"], "dataset_from_index": 0, "dataset_to_index": 10},
            {"tasks": ["b"], "dataset_from_index": 10, "dataset_to_index": 20},
        ]
        first = _build_task_balanced_sampler(episodes, [0, 1], 0, seed=7)
        second = _build_task_balanced_sampler(episodes, [0, 1], 0, seed=7)
        self.assertEqual(list(first), list(second))


if __name__ == "__main__":
    unittest.main()
