"""One-shot handoff from approach motion to the next CALVIN policy retry."""


class RotateApproachState:
    def __init__(self):
        self._pending_skill_id = None

    def arm(self, skill_id):
        self._pending_skill_id = str(skill_id)

    def consume(self, skill_id):
        pending = self._pending_skill_id
        self._pending_skill_id = None
        return pending == str(skill_id)

