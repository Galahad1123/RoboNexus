"""Compose CALVIN oracle localization with XY-threshold recovery motion."""

from .motion_controller import CalvinCartesianApproachController
from .target_localizer import CalvinOracleTargetLocalizer


class CalvinOracleApproachRecovery:
    def __init__(self, adapter):
        self.localizer = CalvinOracleTargetLocalizer(adapter.env)
        self.controller = CalvinCartesianApproachController(adapter)

    def recover(self, episode, observation):
        config = getattr(episode, "retry", {}) or {}
        target_id, target_position, _ = self.localizer.locate(episode)
        return self.controller.execute(
            observation, target_id, target_position, config
        )
