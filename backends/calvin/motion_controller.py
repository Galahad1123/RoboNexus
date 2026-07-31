"""Conservative XY-threshold approach controller for CALVIN."""

import numpy as np

from .schemas import RecoveryResult


class CalvinCartesianApproachController:
    """Move near the target in XY while preserving orientation and gripper."""

    def __init__(self, adapter):
        self.adapter = adapter

    def execute(self, observation, target_id, target_position, config):
        target = np.asarray(target_position, dtype=float)
        threshold = float(config.get("xy_threshold_m", 0.10))
        stop_distance = max(0.0, threshold - min(0.002, threshold * 0.05))
        hover_height = float(config.get("hover_height_m", 0.12))
        tolerance_z = float(config.get("z_tolerance_m", 0.015))
        max_steps = int(config.get("max_motion_steps", 40))
        bounds = config.get(
            "workspace_bounds",
            [[0.05, 0.85], [-0.65, 0.65], [0.45, 1.20]],
        )
        if threshold <= 0:
            raise ValueError("xy_threshold_m must be positive")

        current = np.asarray(observation["robot_obs"][:3], dtype=float)
        initial_xy_distance = float(np.linalg.norm(current[:2] - target[:2]))
        if initial_xy_distance <= threshold:
            return RecoveryResult(
                True,
                observation,
                0,
                "already_within_xy_threshold",
                target_id,
                target,
                initial_xy_distance,
                [],
            )

        safe_z = max(current[2], target[2] + hover_height)
        if not float(bounds[2][0]) <= safe_z <= float(bounds[2][1]):
            return RecoveryResult(
                False,
                observation,
                0,
                "safe_height_outside_workspace",
                target_id,
                target,
                initial_xy_distance,
                [],
            )

        robot = self.adapter.env.robot
        max_delta = float(robot.max_rel_pos) * float(
            getattr(robot, "magic_scaling_factor_pos", 1.0)
        )
        if getattr(robot, "use_target_pose", False):
            robot.target_pos = current.copy()
            robot.target_orn = np.asarray(
                observation["robot_obs"][3:6], dtype=float
            ).copy()

        actions = []
        steps = 0
        gripper = 1.0 if observation["robot_obs"][14] >= 0 else -1.0

        # First lift vertically. Horizontal motion only starts at a safe height.
        while steps < max_steps:
            current = np.asarray(observation["robot_obs"][:3], dtype=float)
            z_error = safe_z - current[2]
            if abs(z_error) <= tolerance_z:
                break
            action = np.zeros(7, dtype=np.float32)
            action[2] = np.clip(z_error / max_delta, -1.0, 1.0)
            action[6] = gripper
            observation, _, done, _ = self.adapter.step(action)
            actions.append(self.adapter.last_executed_action.tolist())
            steps += 1
            if done:
                return RecoveryResult(
                    False,
                    observation,
                    steps,
                    "environment_done",
                    target_id,
                    target,
                    float(
                        np.linalg.norm(
                            np.asarray(observation["robot_obs"][:2]) - target[:2]
                        )
                    ),
                    actions,
                )

        # Then approach horizontally and stop as soon as the threshold is met.
        while steps < max_steps:
            current = np.asarray(observation["robot_obs"][:3], dtype=float)
            xy_delta = target[:2] - current[:2]
            xy_distance = float(np.linalg.norm(xy_delta))
            if xy_distance <= threshold:
                return RecoveryResult(
                    True,
                    observation,
                    steps,
                    "reached_xy_threshold",
                    target_id,
                    target,
                    xy_distance,
                    actions,
                )
            action = np.zeros(7, dtype=np.float32)
            # Do not command all the way to the target: only close the excess
            # distance outside the configured threshold.
            # Aim a few millimetres inside the boundary so float32 action
            # quantization cannot leave the TCP infinitesimally outside it.
            travel = min(max_delta, xy_distance - stop_distance)
            action[:2] = xy_delta / xy_distance * travel / max_delta
            action[6] = gripper
            observation, _, done, _ = self.adapter.step(action)
            actions.append(self.adapter.last_executed_action.tolist())
            steps += 1
            if done:
                return RecoveryResult(
                    False,
                    observation,
                    steps,
                    "environment_done",
                    target_id,
                    target,
                    xy_distance,
                    actions,
                )

        final_xy_distance = float(
            np.linalg.norm(
                np.asarray(observation["robot_obs"][:2], dtype=float) - target[:2]
            )
        )
        return RecoveryResult(
            final_xy_distance <= threshold,
            observation,
            steps,
            "reached_xy_threshold"
            if final_xy_distance <= threshold
            else "motion_timeout",
            target_id,
            target,
            final_xy_distance,
            actions,
        )
