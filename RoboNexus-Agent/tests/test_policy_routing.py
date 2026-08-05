import unittest
from pathlib import Path

from core.policies import (
    HandshakeValidationError,
    PolicyClientPool,
    PolicyRouter,
    PolicyRoutingError,
    load_policy_catalog,
)
from core.skills import CALVIN_D34_ROUTING, load_skill_registry


WORKSPACE = Path(__file__).resolve().parents[2]
AGENT_ROOT = WORKSPACE / "RoboNexus-Agent"


def policy_metadata(**overrides):
    metadata = {
        "protocol_version": 2,
        "policy": "multitask_dit_policy",
        "image_keys": [
            "observation.images.static",
            "observation.images.gripper",
        ],
        "image_shapes": {
            "observation.images.static": [3, 200, 200],
            "observation.images.gripper": [3, 200, 200],
        },
        "state_key": "observation.state",
        "state_dim": 15,
        "action_dim": 7,
        "n_obs_steps": 2,
        "model_action_chunk_size": 16,
        "horizon": 32,
    }
    metadata.update(overrides)
    return metadata


class FakeClient:
    def __init__(self, policy, metadata=None):
        self.policy = policy
        self.metadata = metadata or policy_metadata()
        self.closed = False
        self.health_calls = 0

    def health(self):
        self.health_calls += 1
        return {"type": "health", "ok": True, "metadata": self.metadata}

    def close(self):
        self.closed = True


class PolicyRoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_skill_registry(
            AGENT_ROOT / "configs/calvin/skills_d34.yaml"
        )
        cls.catalog = load_policy_catalog(
            AGENT_ROOT / "configs/calvin/policies.yaml",
            required_policy_ids=cls.registry.policy_groups(),
        )

    def test_router_resolves_all_34_skills_to_canonical_policy(self):
        router = PolicyRouter(self.registry, self.catalog)
        expected = {
            skill_id: policy_id
            for policy_id, skill_ids in CALVIN_D34_ROUTING.items()
            for skill_id in skill_ids
        }
        self.assertEqual(len(expected), 34)
        for skill_id, policy_id in expected.items():
            decision = router.route(skill_id)
            self.assertEqual(decision.skill.skill_id, skill_id)
            self.assertEqual(decision.policy.policy_id, policy_id)

    def test_router_reports_unknown_skill_without_touching_clients(self):
        router = PolicyRouter(self.registry, self.catalog)
        with self.assertRaisesRegex(PolicyRoutingError, "Unknown skill_id"):
            router.route("not_a_calvin_skill")

    def test_catalog_exposes_four_distinct_endpoints(self):
        endpoints = {
            policy_id: self.catalog.get(policy_id).endpoint
            for policy_id in self.catalog.policy_ids()
        }
        self.assertEqual(
            endpoints,
            {
                "general": "ws://127.0.0.1:10093",
                "push": "ws://127.0.0.1:10094",
                "rotate": "ws://127.0.0.1:10095",
                "stack": "ws://127.0.0.1:10096",
            },
        )

    def test_client_pool_lazily_reuses_one_client_per_policy(self):
        created = []

        def factory(policy):
            client = FakeClient(policy)
            created.append(client)
            return client

        pool = PolicyClientPool(self.catalog, client_factory=factory)
        first = pool.get("general")
        second = pool.get("general")
        self.assertIs(first, second)
        self.assertEqual(len(created), 1)
        self.assertEqual(pool.connected_policy_ids, ("general",))
        pool.close_all()
        self.assertTrue(first.closed)

    def test_connect_and_health_check_all_four_policies(self):
        created = {}

        def factory(policy):
            created[policy.policy_id] = FakeClient(policy)
            return created[policy.policy_id]

        with PolicyClientPool(self.catalog, client_factory=factory) as pool:
            results = pool.health_check_all()
            self.assertEqual(set(results), {"general", "push", "rotate", "stack"})
            self.assertEqual(
                pool.connected_policy_ids, ("general", "push", "rotate", "stack")
            )
            self.assertTrue(
                all(client.health_calls == 1 for client in created.values())
            )
        self.assertTrue(all(client.closed for client in created.values()))

    def test_incompatible_handshake_fails_fast_and_closes_new_clients(self):
        created = {}

        def factory(policy):
            metadata = policy_metadata()
            if policy.policy_id == "rotate":
                metadata["action_dim"] = 8
            created[policy.policy_id] = FakeClient(policy, metadata)
            return created[policy.policy_id]

        pool = PolicyClientPool(self.catalog, client_factory=factory)
        with self.assertRaisesRegex(
            HandshakeValidationError, "rotate'[\\s\\S]*action_dim expected 7, got 8"
        ):
            pool.connect_all()
        self.assertEqual(pool.connected_policy_ids, ())
        self.assertTrue(all(client.closed for client in created.values()))

    def test_malformed_handshake_is_rejected(self):
        def factory(policy):
            return FakeClient(policy, {"protocol_version": 2})

        pool = PolicyClientPool(self.catalog, client_factory=factory)
        with self.assertRaisesRegex(HandshakeValidationError, "missing fields"):
            pool.get("general")
        self.assertEqual(pool.connected_policy_ids, ())


if __name__ == "__main__":
    unittest.main()
