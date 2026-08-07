"""Oracle target localization through CALVIN/PyBullet scene state."""

import numpy as np


class CalvinOracleTargetLocalizer:
    """Resolve a configured movable object or fixed link to its world pose."""

    def __init__(self, env):
        self.env = env

    def locate(self, episode):
        recovery = getattr(episode, "retry", {}) or {}
        selector = recovery.get("target")
        if not selector:
            roles = getattr(episode, "mask_roles", {}) or {}
            candidates = roles.get("source") or roles.get("target")
            selector = candidates[0] if candidates else None
        if not selector:
            raise ValueError(
                "No retry target selector configured for {}".format(episode.task)
            )

        return self.locate_selector(selector)

    def locate_grounding(self, grounding, vocabulary):
        if grounding.world_position is not None:
            return (
                grounding.metadata.get("instance_id") or grounding.grounding_id,
                np.asarray(grounding.world_position, dtype=float),
                None,
            )
        semantic_id = grounding.semantic_id
        if not semantic_id:
            raise ValueError(
                "grounding {!r} has no world position or semantic_id".format(
                    grounding.grounding_id
                )
            )
        try:
            selector = dict(vocabulary.selectors[semantic_id])
        except KeyError:
            raise ValueError(
                "semantic_id {!r} is absent from CALVIN vocabulary".format(
                    semantic_id
                )
            )
        if "link" in selector:
            selector["links"] = [selector.pop("link")]
        return self.locate_selector(selector)

    def locate_selector(self, selector):
        scene_info = self.env.get_info()["scene_info"]
        if "movable" in selector:
            name = selector["movable"]
            uid = int(scene_info["movable_objects"][name]["uid"])
            position, orientation = self.env.p.getBasePositionAndOrientation(
                uid, physicsClientId=self.env.cid
            )
            return name, np.asarray(position), np.asarray(orientation)
        if "fixed" in selector:
            fixed_name = selector["fixed"]
            info = scene_info["fixed_objects"][fixed_name]
            uid = int(info["uid"])
            links = selector.get("links") or []
            if links:
                link_name = links[0]
                link_id = int(info["links"][link_name])
                state = self.env.p.getLinkState(
                    uid, link_id, physicsClientId=self.env.cid
                )
                return (
                    "{}:{}".format(fixed_name, link_name),
                    np.asarray(state[0]),
                    np.asarray(state[1]),
                )
            position, orientation = self.env.p.getBasePositionAndOrientation(
                uid, physicsClientId=self.env.cid
            )
            return fixed_name, np.asarray(position), np.asarray(orientation)
        raise ValueError("Unsupported retry target selector: {}".format(selector))
