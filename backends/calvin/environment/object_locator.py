"""Replaceable CALVIN object-location inspection backend."""


BLOCK_NAMES = {
    "block_red": "red_block",
    "block_blue": "blue_block",
    "block_pink": "pink_block",
}


class CalvinOracleObjectLocator:
    """Classify block support from CALVIN contact metadata.

    The Agent-facing output is deliberately a small semantic mapping so this
    implementation can later be replaced by a VLM/tracker or a real-robot
    perception backend without changing the inspect_scene tool contract.
    """

    def __init__(self, adapter):
        self.adapter = adapter

    def locate_objects(self):
        info = self.adapter.get_info()
        scene = info["scene_info"]
        movable = scene["movable_objects"]
        robot_uid = int(info["robot_info"]["uid"])
        fixed_by_uid = {
            int(item["uid"]): dict(item.get("links", {}))
            for item in scene["fixed_objects"].values()
        }
        result = {}
        for backend_name, public_name in BLOCK_NAMES.items():
            obj = movable.get(backend_name)
            result[public_name] = (
                self._location(obj, robot_uid, fixed_by_uid)
                if isinstance(obj, dict)
                else "unknown"
            )
        return result

    @staticmethod
    def _location(obj, robot_uid, fixed_by_uid):
        supports = set()
        held = False
        for contact in obj.get("contacts", ()):
            body_uid = int(contact[2])
            link_index = int(contact[4])
            if body_uid == robot_uid:
                held = True
                continue
            links = fixed_by_uid.get(body_uid)
            if links is None:
                continue
            link_name = next(
                (name for name, index in links.items() if int(index) == link_index),
                "base_link" if link_index == -1 else str(link_index),
            )
            supports.add(link_name)
        if held:
            return "gripper"
        if "drawer_link" in supports:
            return "drawer"
        if "plank_link" in supports:
            return "slider"
        if "base_link" in supports:
            return "table"
        return "unknown"
