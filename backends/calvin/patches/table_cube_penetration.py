"""Detect and recover CALVIN cubes pushed through the tabletop."""

from dataclasses import dataclass
import re

import numpy as np


ELIGIBLE_SKILL = re.compile(
    r"^(lift_(red|blue|pink)_block_table|"
    r"push_(red|blue|pink)_block_(left|right)|"
    r"rotate_(red|blue|pink)_block_(left|right))$"
)
BLOCK_SEMANTICS = {
    "block_red": "block_red",
    "red_block": "block_red",
    "block_blue": "block_blue",
    "blue_block": "block_blue",
    "block_pink": "block_pink",
    "pink_block": "block_pink",
}


@dataclass(frozen=True)
class CubeState:
    position: tuple
    orientation: tuple
    linear_velocity: tuple
    angular_velocity: tuple


class TableCubePenetrationDetector:
    """Read-only during valid motion; mutates simulation only on recovery."""

    def __init__(
        self,
        adapter,
        skill_id,
        object_name,
        initial_info,
        penetration_threshold_m=0.008,
        confirmation_steps=2,
        settle_steps=16,
        table_xy_margin_m=0.12,
    ):
        self.adapter = adapter
        self.skill_id = skill_id
        self.object_name = object_name
        self.penetration_threshold_m = float(penetration_threshold_m)
        self.confirmation_steps = int(confirmation_steps)
        self.settle_steps = int(settle_steps)
        self.table_xy_bounds = _table_xy_bounds(adapter, float(table_xy_margin_m))
        initial_state = (
            _cube_state(initial_info, object_name) if object_name else None
        )
        eligible = bool(
            ELIGIBLE_SKILL.match(skill_id)
            and initial_state is not None
            and _inside_xy(initial_state.position, self.table_xy_bounds)
        )
        supported = _is_supported_by_table(initial_info, object_name)
        held = _is_held_by_robot(initial_info, object_name)
        # A push may legitimately start with the source still held after a
        # recovery lift. Keep monitoring, but arm only after the cube first
        # establishes a valid tabletop contact. Objects starting on another
        # facility remain outside this CALVIN patch's scope.
        self.enabled = bool(eligible and (supported or held))
        self._armed = bool(self.enabled and supported)
        self._consecutive = 0
        self._last_valid = initial_state if self._armed else None
        self._reference_z = (
            float(self._last_valid.position[2]) if self._last_valid is not None else None
        )
        self._anomaly = None

    @classmethod
    def for_attempt(cls, adapter, skill_id, groundings, initial_info):
        object_name = None
        for grounding in groundings:
            if grounding.role.value not in {"source", "target"}:
                continue
            object_name = BLOCK_SEMANTICS.get(grounding.semantic_id)
            if object_name is not None:
                break
        return cls(adapter, skill_id, object_name, initial_info) if object_name else None

    def observe(self, current_info):
        if not self.enabled or self._anomaly is not None:
            return self._anomaly
        state = _cube_state(current_info, self.object_name)
        if not _inside_xy(state.position, self.table_xy_bounds):
            # Falling beyond the tabletop edge is a real scene outcome, not
            # penetration through the tabletop. Never rewind it as a patch.
            self.enabled = False
            self._consecutive = 0
            return None
        if not self._armed:
            if not _is_supported_by_table(current_info, self.object_name):
                return None
            self._armed = True
            self._last_valid = state
            self._reference_z = float(state.position[2])
            self._consecutive = 0
            return None
        penetration = self._reference_z - float(state.position[2])
        if penetration > self.penetration_threshold_m:
            self._consecutive += 1
            if self._consecutive >= self.confirmation_steps:
                self._anomaly = {
                    "type": "table_cube_penetration",
                    "skill_id": self.skill_id,
                    "object_name": self.object_name,
                    "reference_z": self._reference_z,
                    "detected_z": float(state.position[2]),
                    "penetration_m": penetration,
                    "threshold_m": self.penetration_threshold_m,
                    "confirmation_steps": self._consecutive,
                    "table_xy_bounds": self.table_xy_bounds,
                }
                return dict(self._anomaly)
            return None
        self._consecutive = 0
        self._last_valid = state
        return None

    def recover(self):
        if self._anomaly is None or self._last_valid is None:
            return {"performed": False, "reason": "no_penetration_detected"}
        env = self.adapter.env
        info = self.adapter.get_info()
        uid = int(info["scene_info"]["movable_objects"][self.object_name]["uid"])
        state = self._last_valid
        env.p.resetBasePositionAndOrientation(
            uid,
            state.position,
            state.orientation,
            physicsClientId=env.cid,
        )
        env.p.resetBaseVelocity(
            uid,
            linearVelocity=(0.0, 0.0, 0.0),
            angularVelocity=(0.0, 0.0, 0.0),
            physicsClientId=env.cid,
        )
        for _ in range(self.settle_steps):
            env.p.stepSimulation(physicsClientId=env.cid)
        self.adapter.current_observation = env.get_obs()
        result = dict(self._anomaly)
        result.update(
            {
                "performed": True,
                "restored_position": list(state.position),
                "velocities_cleared": True,
                "settle_steps": self.settle_steps,
            }
        )
        return result


def _cube_state(info, object_name):
    obj = info["scene_info"]["movable_objects"][object_name]
    values = (
        obj["current_pos"],
        obj["current_orn"],
        obj.get("current_lin_vel", (0.0, 0.0, 0.0)),
        obj.get("current_ang_vel", (0.0, 0.0, 0.0)),
    )
    arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
    if [array.size for array in arrays] != [3, 4, 3, 3]:
        raise ValueError("invalid CALVIN cube state")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("CALVIN cube state must be finite")
    return CubeState(*(tuple(float(item) for item in array) for array in arrays))


def _is_supported_by_table(info, object_name):
    if not object_name:
        return False
    scene = info["scene_info"]
    obj = scene["movable_objects"].get(object_name)
    if not isinstance(obj, dict):
        return False
    fixed_by_uid = {
        int(item["uid"]): dict(item.get("links", {}))
        for item in scene["fixed_objects"].values()
    }
    robot_uid = int(info["robot_info"]["uid"])
    for contact in obj.get("contacts", ()):
        body_uid, link_index = int(contact[2]), int(contact[4])
        if body_uid == robot_uid or body_uid not in fixed_by_uid:
            continue
        links = fixed_by_uid[body_uid]
        link_name = next(
            (name for name, index in links.items() if int(index) == link_index),
            "base_link" if link_index == -1 else None,
        )
        if link_name == "base_link":
            return True
    return False


def _is_held_by_robot(info, object_name):
    if not object_name:
        return False
    scene = info["scene_info"]
    obj = scene["movable_objects"].get(object_name)
    if not isinstance(obj, dict):
        return False
    robot_uid = int(info["robot_info"]["uid"])
    return any(int(contact[2]) == robot_uid for contact in obj.get("contacts", ()))


def _table_xy_bounds(adapter, margin):
    try:
        surface = np.asarray(
            adapter.env.scene.surfaces["table"], dtype=np.float64
        )
        if surface.shape != (2, 3) or not np.isfinite(surface).all():
            raise ValueError("invalid table surface")
        low = np.minimum(surface[0, :2], surface[1, :2]) - margin
        high = np.maximum(surface[0, :2], surface[1, :2]) + margin
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        # CALVIN-D table sampling range [0,-.15]..[.35,-.03] with a
        # conservative 12 cm allowance for legitimate lateral pushes.
        low = np.asarray((0.0, -0.15), dtype=np.float64) - margin
        high = np.asarray((0.35, -0.03), dtype=np.float64) + margin
    return {
        "x": [float(low[0]), float(high[0])],
        "y": [float(low[1]), float(high[1])],
    }


def _inside_xy(position, bounds):
    x, y = float(position[0]), float(position[1])
    return (
        bounds["x"][0] <= x <= bounds["x"][1]
        and bounds["y"][0] <= y <= bounds["y"][1]
    )
