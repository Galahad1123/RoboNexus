"""PyBullet object/link segmentation provider for CALVIN."""

import numpy as np

from .base import MaskProvider


class CalvinGTMaskProvider(MaskProvider):
    def __init__(self, env):
        self.env = env
        self._resolved_roles = None

    def reset(self, episode):
        super().reset(episode)
        self._resolved_roles = None

    def masks(self, observation):
        roles = getattr(self.episode, "mask_roles", {}) or {}
        if not roles:
            return {}
        if self._resolved_roles is None:
            scene_info = self.env.get_info()["scene_info"]
            self._resolved_roles = {
                role: self._resolve_selectors(selectors, scene_info)
                for role, selectors in roles.items()
            }
        result = {}
        for camera in self.env.cameras:
            camera_key = "observation.images.{}".format(camera.name)
            segmentation = self._render_segmentation(camera)
            result[camera_key] = {
                role: self._select(segmentation, identities)
                for role, identities in self._resolved_roles.items()
            }
        return result

    def _resolve_selectors(self, selectors, scene_info):
        identities = set()
        held_uids = self._held_movable_uids(scene_info)
        movable = scene_info["movable_objects"]
        fixed = scene_info["fixed_objects"]
        for selector in selectors:
            if "movable" in selector:
                name = selector["movable"]
                names = movable.keys() if name == "*" else (name,)
                identities.update((int(movable[item]["uid"]), None) for item in names)
            elif selector.get("held_movable"):
                identities.update((uid, None) for uid in held_uids)
            elif selector.get("unheld_movable"):
                identities.update(
                    (int(info["uid"]), None)
                    for info in movable.values()
                    if int(info["uid"]) not in held_uids
                )
            elif "fixed" in selector:
                info = fixed[selector["fixed"]]
                links = selector.get("links")
                if links:
                    identities.update(
                        (int(info["uid"]), int(info["links"][link])) for link in links
                    )
                else:
                    identities.add((int(info["uid"]), None))
            else:
                raise ValueError("Unsupported mask selector: {}".format(selector))
        return identities

    def _held_movable_uids(self, scene_info):
        robot_uid = int(self.env.robot.robot_uid)
        held = set()
        for info in scene_info["movable_objects"].values():
            uid = int(info["uid"])
            contacts = self.env.p.getContactPoints(
                bodyA=robot_uid, bodyB=uid, physicsClientId=self.env.cid
            )
            if contacts:
                held.add(uid)
        return held

    def _render_segmentation(self, camera):
        flags = self.env.p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        if hasattr(camera, "viewMatrix"):
            view, projection = camera.viewMatrix, camera.projectionMatrix
        else:
            # GripperCamera computes these during the observation render.
            view, projection = camera.view_matrix, camera.projection_matrix
        image = self.env.p.getCameraImage(
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
    def _select(segmentation, identities):
        valid = segmentation >= 0
        body_uid = segmentation & ((1 << 24) - 1)
        link_index = (segmentation >> 24) - 1
        result = np.zeros(segmentation.shape, dtype=bool)
        for uid, link in identities:
            selected = valid & (body_uid == uid)
            if link is not None:
                selected &= link_index == link
            result |= selected
        return result
