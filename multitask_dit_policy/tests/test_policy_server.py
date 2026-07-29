from collections import deque
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from lerobot.utils.constants import ACTION

from multitask_dit_policy.serving.inference_session import InferenceSession


class FakePolicy:
    def __init__(self):
        feature = lambda shape: SimpleNamespace(shape=shape, type="unused")
        self.config = SimpleNamespace(
            n_action_steps=4,
            n_obs_steps=2,
            horizon=8,
            image_features=[
                "observation.images.static",
                "observation.images.gripper",
            ],
            robot_state_feature=feature((15,)),
            action_feature=feature((7,)),
            input_features={
                "observation.state": feature((15,)),
                "observation.images.static": feature((3, 4, 4)),
                "observation.images.gripper": feature((3, 4, 4)),
            },
            output_features={ACTION: feature((7,))},
            normalization_mapping={},
        )
        self.generated_chunks = 0
        self.reset()

    def reset(self):
        self._queues = {
            "observation.state": deque(maxlen=2),
            "observation.images": deque(maxlen=2),
            "task": deque(maxlen=2),
            ACTION: deque(maxlen=4),
        }

    def select_action(self, batch):
        self._queues["observation.state"].append(batch["observation.state"])
        if not self._queues[ACTION]:
            self.generated_chunks += 1
            value = float(self.generated_chunks)
            self._queues[ACTION].extend(
                torch.full((1, 7), value) for _ in range(self.config.n_action_steps)
            )
        return self._queues[ACTION].popleft()


def observation(value=0):
    return {
        "observation.state": np.full(15, value, dtype=np.float32),
        "observation.images.static": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation.images.gripper": np.zeros((4, 4, 3), dtype=np.uint8),
    }


class InferenceSessionTest(unittest.TestCase):
    def test_replans_after_configured_number_of_actions(self):
        policy = FakePolicy()
        session = InferenceSession(
            policy=policy,
            stats_tensors={},
            device=torch.device("cpu"),
        )
        session.reset(task="test", episode_id="ep", replan_interval=2)

        outputs = [session.step(observation(index))[0][0] for index in range(5)]

        self.assertEqual(outputs, [1.0, 1.0, 2.0, 2.0, 3.0])
        self.assertEqual(policy.generated_chunks, 3)
        self.assertEqual(len(policy._queues["observation.state"]), 2)

    def test_replan_interval_is_validated(self):
        session = InferenceSession(
            policy=FakePolicy(),
            stats_tensors={},
            device=torch.device("cpu"),
        )
        with self.assertRaisesRegex(ValueError, "between 1 and 4"):
            session.reset(replan_interval=5)

    def test_rejects_wrong_image_resolution(self):
        session = InferenceSession(
            policy=FakePolicy(),
            stats_tensors={},
            device=torch.device("cpu"),
        )
        bad = observation()
        bad["observation.images.gripper"] = np.zeros((3, 3, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "resolution"):
            session.step(bad)

    def test_episode_seed_resets_torch_rng(self):
        session = InferenceSession(
            policy=FakePolicy(),
            stats_tensors={},
            device=torch.device("cpu"),
        )
        session.reset(episode_seed=1234)
        first = torch.rand(4)
        session.reset(episode_seed=1234)
        second = torch.rand(4)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
