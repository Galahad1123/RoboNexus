import unittest

from multitask_dit_policy.train import _split_episode_indices


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


if __name__ == "__main__":
    unittest.main()
