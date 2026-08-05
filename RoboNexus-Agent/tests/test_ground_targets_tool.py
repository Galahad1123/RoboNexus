import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backends.calvin.grounding.oracle_grounder import CalvinOracleGrounder
from backends.calvin.grounding.semantic_vocabulary import load_semantic_vocabulary
from backends.local_artifact_store import LocalArtifactStore
from core.domain import (
    ArtifactRef,
    ErrorCode,
    GoalInput,
    GroundedObject,
    GroundingRepresentation,
    ObservationRef,
    Role,
    ToolStatus,
)
from core.ports import (
    GrounderPort,
    GroundingBatch,
    GroundingNotFoundError,
    UnsupportedGroundingError,
)
from core.runtime import AgentSession
from core.tools import GroundTargetsTool


def make_session():
    initial = ObservationRef(
        "obs_000001", 0, 0, "artifact://initial/static.png", "artifact://initial/wrist.png"
    )
    return AgentSession(
        GoalInput(
            "goal-ground",
            "stack the red block on the blue block",
            {"predicates": ["stack_block"]},
            initial,
        ),
        session_id="session-ground",
    )


def semantic_params(observation_id="obs_000001", semantic_id="block_red"):
    return {
        "skill_id": "stack_block",
        "observation_id": observation_id,
        "proposals": [
            {
                "role": "source",
                "representation": "semantic",
                "semantic_id": semantic_id,
                "description": "the red block selected by the planner",
            }
        ],
    }


class FakeGrounder(GrounderPort):
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def capabilities(self):
        return {"representations": ("semantic",)}

    def ground(self, skill_id, observation, proposals):
        self.calls += 1
        if self.error:
            raise self.error
        groundings = []
        artifacts = []
        for proposal in proposals:
            handle = "mask://session/{}.npy".format(proposal["grounding_id"])
            artifacts.append(
                ArtifactRef(
                    proposal["grounding_id"], handle, "application/x-numpy-bool-mask"
                )
            )
            groundings.append(
                GroundedObject(
                    proposal["grounding_id"],
                    Role(proposal["role"]),
                    "fake",
                    observation.observation_id,
                    observation.environment_revision,
                    GroundingRepresentation.SEMANTIC,
                    semantic_id=proposal.get("semantic_id"),
                    camera="static",
                    mask_handle=handle,
                    confidence=1.0,
                )
            )
        preview = ArtifactRef(
            "grounding_preview",
            "artifact://session/preview.png",
            "image/png",
        )
        return GroundingBatch(tuple(groundings), preview, tuple(artifacts))


class GroundTargetsToolTest(unittest.TestCase):
    def test_success_registers_grounding_and_preview(self):
        session = make_session()
        result = GroundTargetsTool(
            session, {"stack_block"}, FakeGrounder()
        ).call(semantic_params(), tool_call_id="ground-call")
        self.assertEqual(result.status, ToolStatus.SUCCEEDED)
        self.assertEqual(result.data["skill_id"], "stack_block")
        self.assertEqual(result.data["groundings"][0]["role"], "source")
        self.assertEqual(result.data["groundings"][0]["semantic_id"], "block_red")
        self.assertTrue(result.data["groundings"][0]["mask_handle"].startswith("mask://"))
        self.assertEqual(len(result.artifacts), 2)
        grounding_id = result.data["groundings"][0]["grounding_id"]
        self.assertEqual(session.get_grounding(grounding_id).source, "fake")

    def test_two_roles_are_preserved(self):
        params = semantic_params()
        params["proposals"].append(
            {
                "role": "destination",
                "representation": "semantic",
                "semantic_id": "block_blue",
                "description": "the blue destination block",
            }
        )
        result = GroundTargetsTool(
            make_session(), {"stack_block"}, FakeGrounder()
        ).call(params)
        self.assertEqual(result.status, ToolStatus.SUCCEEDED)
        self.assertEqual(
            [item["role"] for item in result.data["groundings"]],
            ["source", "destination"],
        )

    def test_replay_does_not_ground_twice(self):
        grounder = FakeGrounder()
        session = make_session()
        tool = GroundTargetsTool(session, {"stack_block"}, grounder)
        first = tool.call(semantic_params(), tool_call_id="same")
        second = tool.call(semantic_params(semantic_id="block_blue"), tool_call_id="same")
        self.assertIs(first, second)
        self.assertEqual(grounder.calls, 1)
        self.assertEqual(session.tool_calls_used, 1)

    def test_unknown_skill_is_rejected(self):
        params = semantic_params()
        params["skill_id"] = "not_a_skill"
        result = GroundTargetsTool(
            make_session(), {"stack_block"}, FakeGrounder()
        ).call(params)
        self.assertEqual(result.status, ToolStatus.REJECTED)
        self.assertEqual(result.error.code, ErrorCode.UNKNOWN_SKILL)

    def test_stale_observation_is_rejected_before_grounder(self):
        session = make_session()
        session.add_environment_steps(1)
        grounder = FakeGrounder()
        result = GroundTargetsTool(
            session, {"stack_block"}, grounder
        ).call(semantic_params())
        self.assertEqual(result.error.code, ErrorCode.STALE_OBSERVATION)
        self.assertEqual(grounder.calls, 0)

    def test_cross_field_validation_rejects_missing_semantic_id(self):
        params = semantic_params()
        del params["proposals"][0]["semantic_id"]
        result = GroundTargetsTool(
            make_session(), {"stack_block"}, FakeGrounder()
        ).call(params)
        self.assertEqual(result.error.code, ErrorCode.INVALID_ARGUMENT)

    def test_duplicate_roles_are_rejected(self):
        params = semantic_params()
        params["proposals"].append(dict(params["proposals"][0]))
        result = GroundTargetsTool(
            make_session(), {"stack_block"}, FakeGrounder()
        ).call(params)
        self.assertEqual(result.error.code, ErrorCode.INVALID_ARGUMENT)

    def test_backend_errors_are_structured(self):
        cases = (
            (UnsupportedGroundingError("bbox unavailable"), ErrorCode.UNSUPPORTED_GROUNDING),
            (GroundingNotFoundError("empty mask"), ErrorCode.GROUNDING_NOT_FOUND),
        )
        for error, code in cases:
            result = GroundTargetsTool(
                make_session(), {"stack_block"}, FakeGrounder(error)
            ).call(semantic_params())
            self.assertEqual(result.status, ToolStatus.REJECTED)
            self.assertEqual(result.error.code, code)

    def test_local_artifact_store_round_trips_boolean_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            original = np.array([[False, True], [True, False]])
            artifact = store.save_mask("session", "ground_1", original)
            self.assertTrue(artifact.uri.startswith("mask://"))
            loaded = np.load(str(store.resolve(artifact.uri)), allow_pickle=False)
            np.testing.assert_array_equal(loaded, original)


class FakeOverlay:
    def apply(self, image, role_masks):
        result = np.asarray(image).copy()
        for mask in role_masks.values():
            result[mask] = (255, 0, 0)
        return result


class FakeArtifacts:
    def __init__(self):
        self.masks = []

    def save_mask(self, session_id, kind, mask):
        self.masks.append(np.asarray(mask).copy())
        return ArtifactRef(kind, "mask://{}/{}.npy".format(session_id, kind), "mask/bool")

    def save_image(self, session_id, kind, image):
        return ArtifactRef(kind, "artifact://{}/{}.png".format(session_id, kind), "image/png")


class FakePhysics:
    ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX = 1

    def __init__(self, segmentation):
        self.segmentation = segmentation

    def getCameraImage(self, **kwargs):
        return (None, None, None, None, self.segmentation)


class CalvinOracleGrounderTest(unittest.TestCase):
    def test_semantic_object_and_link_grounding(self):
        object_uid = 7
        table_uid = 9
        drawer_link = 3
        segmentation = np.full((4, 5), -1, dtype=np.int64)
        segmentation[0:2, 0:2] = object_uid
        segmentation[2:4, 2:5] = table_uid + ((drawer_link + 1) << 24)
        camera = type(
            "Camera",
            (),
            {
                "name": "static",
                "width": 5,
                "height": 4,
                "viewMatrix": (),
                "projectionMatrix": (),
                "cid": 0,
            },
        )()
        physics = FakePhysics(segmentation)
        scene_info = {
            "movable_objects": {"block_red": {"uid": object_uid}},
            "fixed_objects": {
                "table": {"uid": table_uid, "links": {"drawer_link": drawer_link}}
            },
        }
        env = type(
            "Env",
            (),
            {
                "cameras": [camera],
                "p": physics,
                "cid": 0,
                "get_info": lambda self: {"scene_info": scene_info},
            },
        )()
        adapter = type(
            "Adapter",
            (),
            {
                "env": env,
                "current_observation": {
                    "rgb_obs": {"rgb_static": np.zeros((4, 5, 3), dtype=np.uint8)}
                },
            },
        )()
        observation = make_session().latest_observation
        proposals = [
            {
                "grounding_id": "ground_1",
                "role": "source",
                "representation": "semantic",
                "semantic_id": "block_red",
                "description": "red block",
            },
            {
                "grounding_id": "ground_2",
                "role": "destination",
                "representation": "semantic",
                "semantic_id": "drawer",
                "description": "drawer",
            },
        ]
        artifacts = FakeArtifacts()
        batch = CalvinOracleGrounder(
            adapter, artifacts, "session", overlay=FakeOverlay()
        ).ground("push_into_drawer", observation, proposals)
        self.assertEqual(len(batch.groundings), 2)
        self.assertEqual(int(artifacts.masks[0].sum()), 4)
        self.assertEqual(int(artifacts.masks[1].sum()), 6)

    def test_capabilities_publish_protocol_vocabulary(self):
        grounder = CalvinOracleGrounder(None, None, "session")
        capabilities = grounder.capabilities()
        self.assertEqual(capabilities["representations"], ("semantic",))
        self.assertIn("block_pink", capabilities["semantic_vocabulary"])
        self.assertIn("light_switch", capabilities["semantic_vocabulary"])
        self.assertEqual(capabilities["calvin_splits"], ("A", "B", "C", "D"))

    def test_custom_vocabulary_path_extends_without_code_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocabulary.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "name": "calvin_ab_experiment",
                        "calvin_splits": ["A", "B"],
                        "semantics": {
                            "custom_marker": {"fixed": "table", "link": "base_link"}
                        },
                    }
                )
            )
            grounder = CalvinOracleGrounder(
                None, None, "session", vocabulary_path=path
            )
            capabilities = grounder.capabilities()
            self.assertEqual(capabilities["semantic_vocabulary"], ("custom_marker",))
            self.assertEqual(capabilities["vocabulary_name"], "calvin_ab_experiment")

    def test_invalid_vocabulary_selector_is_rejected_at_load_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "name": "bad",
                        "calvin_splits": ["D"],
                        "semantics": {"broken": {"fixed": "table"}},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "fixed and link"):
                load_semantic_vocabulary(path)


if __name__ == "__main__":
    unittest.main()
