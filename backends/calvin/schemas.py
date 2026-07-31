"""Internal result types for the Python-3.8 CALVIN backend."""

from dataclasses import dataclass

import numpy as np


@dataclass
class RecoveryResult:
    success: bool
    observation: object
    steps: int
    reason: str
    target_id: str = None
    target_position: object = None
    final_xy_distance_m: float = None
    actions: object = None

    def as_dict(self):
        return {
            "success": bool(self.success),
            "steps": int(self.steps),
            "reason": self.reason,
            "target_id": self.target_id,
            "target_position": (
                np.asarray(self.target_position).tolist()
                if self.target_position is not None
                else None
            ),
            "final_xy_distance_m": self.final_xy_distance_m,
        }
