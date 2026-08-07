"""CALVIN Oracle semantic grounding backed by PyBullet segmentation."""

import numpy as np

from calvin.evaluation.remote_policy.masking.overlay import MaskOverlay
from core.domain import GroundedObject, GroundingRepresentation, Role
from core.ports import (
    GrounderPort,
    GroundingBatch,
    GroundingNotFoundError,
    UnsupportedGroundingError,
)

from .semantic_vocabulary import load_semantic_vocabulary


class CalvinOracleGrounder(GrounderPort):
    """Resolve planner-selected semantic IDs without selecting on its behalf."""

    def __init__(
        self,
        adapter,
        artifacts,
        session_id,
        overlay=None,
        vocabulary=None,
        vocabulary_path=None,
    ):
        if vocabulary is not None and vocabulary_path is not None:
            raise ValueError("provide vocabulary or vocabulary_path, not both")
        self.adapter = adapter
        self.artifacts = artifacts
        self.session_id = session_id
        self.vocabulary = vocabulary or load_semantic_vocabulary(vocabulary_path)
        self.overlay = overlay or MaskOverlay(
            alpha=0.25,
            outline_width_px=2,
            outline_color=(0, 0, 0),
            outline_position="center",
        )

    def capabilities(self):
        return {
            "representations": ("semantic",),
            "semantic_vocabulary": self.vocabulary.semantic_ids(),
            "vocabulary_name": self.vocabulary.name,
            "vocabulary_version": self.vocabulary.version,
            "calvin_splits": self.vocabulary.calvin_splits,
            "cameras": ("static",),
            "source": "calvin_gt",
        }

    def ground(self, skill_id, observation, proposals):
        del skill_id
        current = getattr(self.adapter, "current_observation", None)
        if not isinstance(current, dict):
            raise GroundingNotFoundError(
                "CALVIN environment has no current observation; reset it first"
            )
        try:
            static_rgb = np.asarray(
                current["rgb_obs"]["rgb_static"], dtype=np.uint8
            ).copy()
            scene_info = self.adapter.env.get_info()["scene_info"]
        except (KeyError, TypeError, ValueError) as exc:
            raise GroundingNotFoundError(
                "invalid CALVIN scene for grounding: {}".format(exc)
            )
        segmentation = self._render_static_segmentation()
        if segmentation.shape != static_rgb.shape[:2]:
            raise GroundingNotFoundError(
                "static RGB and segmentation dimensions do not match"
            )

        grounded = []
        mask_artifacts = []
        role_masks = {}
        for proposal in proposals:
            if proposal["representation"] != "semantic":
                raise UnsupportedGroundingError(
                    "CALVIN Oracle grounder supports semantic proposals only"
                )
            semantic_id = proposal["semantic_id"]
            try:
                selector = self.vocabulary.selectors[semantic_id]
            except KeyError:
                raise UnsupportedGroundingError(
                    "unsupported CALVIN semantic_id {!r}".format(semantic_id)
                )
            identity = self._resolve_identity(selector, scene_info)
            mask = self._select(segmentation, identity)
            if not bool(mask.any()):
                raise GroundingNotFoundError(
                    "semantic target {!r} is not visible in static camera".format(
                        semantic_id
                    )
                )
            mask_artifact = self.artifacts.save_mask(
                self.session_id, proposal["grounding_id"], mask
            )
            mask_artifact.validate()
            role = Role(proposal["role"])
            grounding = GroundedObject(
                grounding_id=proposal["grounding_id"],
                role=role,
                source="calvin_gt",
                observation_id=observation.observation_id,
                environment_revision=observation.environment_revision,
                representation=GroundingRepresentation.SEMANTIC,
                semantic_id=semantic_id,
                camera="static",
                mask_handle=mask_artifact.uri,
                confidence=1.0,
                metadata={
                    "description": proposal["description"],
                    "instance_id": _instance_id(selector),
                },
            )
            grounding.validate()
            grounded.append(grounding)
            mask_artifacts.append(mask_artifact)
            role_masks[role.value] = mask

        preview_rgb = self.overlay.apply(static_rgb, role_masks)
        preview = self.artifacts.save_image(
            self.session_id, "grounding_preview", preview_rgb
        )
        preview.validate()
        return GroundingBatch(tuple(grounded), preview, tuple(mask_artifacts))

    def _render_static_segmentation(self):
        cameras = getattr(self.adapter.env, "cameras", ())
        camera = next(
            (
                item
                for item in cameras
                if str(getattr(item, "name", "")).lower()
                in {"static", "rgb_static"}
            ),
            None,
        )
        if camera is None:
            raise GroundingNotFoundError("CALVIN static camera is unavailable")
        flags = self.adapter.env.p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        if hasattr(camera, "viewMatrix"):
            view, projection = camera.viewMatrix, camera.projectionMatrix
        else:
            view = camera.view_matrix
            projection = camera.projection_matrix
        image = self.adapter.env.p.getCameraImage(
            width=camera.width,
            height=camera.height,
            viewMatrix=view,
            projectionMatrix=projection,
            flags=flags,
            physicsClientId=camera.cid,
        )
        return np.asarray(image[4], dtype=np.int64).reshape(
            camera.height, camera.width
        )

    @staticmethod
    def _resolve_identity(selector, scene_info):
        if "movable" in selector:
            try:
                return int(scene_info["movable_objects"][selector["movable"]]["uid"]), None
            except KeyError as exc:
                raise GroundingNotFoundError(
                    "CALVIN movable object is unavailable: {}".format(exc)
                )
        try:
            fixed = scene_info["fixed_objects"][selector["fixed"]]
            return int(fixed["uid"]), int(fixed["links"][selector["link"]])
        except KeyError as exc:
            raise GroundingNotFoundError(
                "CALVIN fixed object link is unavailable: {}".format(exc)
            )

    @staticmethod
    def _select(segmentation, identity):
        uid, link = identity
        valid = segmentation >= 0
        body_uid = segmentation & ((1 << 24) - 1)
        selected = valid & (body_uid == uid)
        if link is not None:
            link_index = (segmentation >> 24) - 1
            selected &= link_index == link
        return selected


def _instance_id(selector):
    if "movable" in selector:
        return "movable:{}".format(selector["movable"])
    return "fixed:{}:{}".format(selector["fixed"], selector["link"])
