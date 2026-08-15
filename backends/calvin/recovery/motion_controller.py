"""Conservative XY-threshold approach controller for CALVIN."""

import numpy as np

from .schemas import RecoveryResult


class CalvinCartesianApproachController:
    """Move near the target with an optional fixed top-down orientation."""

    def __init__(self, adapter):
        self.adapter = adapter

    def execute(self, observation, target_id, target_position, config):
        target = np.asarray(target_position, dtype=float)
        threshold = float(config.get("xy_threshold_m", 0.10))
        stop_distance = max(0.0, threshold - min(0.002, threshold * 0.05))
        hover_height = float(config.get("hover_height_m", 0.12))
        tolerance_z = float(config.get("z_tolerance_m", 0.015))
        orientation_tolerance = float(
            config.get("orientation_tolerance_rad", 0.08)
        )
        target_orientation = config.get("target_orientation_euler")
        target_orientation = (
            None
            if target_orientation is None
            else np.asarray(target_orientation, dtype=float).reshape(3)
        )
        max_steps = int(config.get("max_motion_steps", 40))
        bounds = config.get(
            "workspace_bounds",
            [[0.05, 0.85], [-0.65, 0.65], [0.45, 1.20]],
        )
        if threshold <= 0:
            raise ValueError("xy_threshold_m must be positive")

        current = np.asarray(observation["robot_obs"][:3], dtype=float)
        initial_xy_distance = float(np.linalg.norm(current[:2] - target[:2]))
        if initial_xy_distance <= threshold and target_orientation is None:
            return RecoveryResult(
                True,
                observation,
                0,
                "already_within_xy_threshold",
                target_id,
                target,
                initial_xy_distance,
                [],
                0.0,
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
        # The rotate-recovery adapter explicitly requests an open gripper;
        # generic/legacy callers retain their previous gripper command.
        gripper = float(
            config.get(
                "gripper_command",
                1.0 if observation["robot_obs"][14] >= 0 else -1.0,
            )
        )
        max_orn_delta = float(getattr(robot, "max_rel_orn", 0.05)) * float(
            getattr(robot, "magic_scaling_factor_orn", 1.0)
        )

        def orientation_error(current_observation):
            if target_orientation is None:
                return np.zeros(3, dtype=float)
            current_orientation = np.asarray(
                current_observation["robot_obs"][3:6], dtype=float
            )
            return _wrapped_angle_delta(current_orientation, target_orientation)

        def fill_orientation(action, current_observation):
            error = orientation_error(current_observation)
            if target_orientation is not None:
                action[3:6] = np.clip(
                    error / max_orn_delta, -1.0, 1.0
                )
            return float(np.max(np.abs(error)))

        # First lift vertically. Horizontal motion only starts at a safe height.
        while steps < max_steps:
            current = np.asarray(observation["robot_obs"][:3], dtype=float)
            z_error = safe_z - current[2]
            orn_error = float(np.max(np.abs(orientation_error(observation))))
            if abs(z_error) <= tolerance_z and orn_error <= orientation_tolerance:
                break
            action = np.zeros(7, dtype=np.float32)
            action[2] = np.clip(z_error / max_delta, -1.0, 1.0)
            fill_orientation(action, observation)
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
                    float(np.max(np.abs(orientation_error(observation)))),
                )

        # Then approach horizontally and stop as soon as the threshold is met.
        while steps < max_steps:
            current = np.asarray(observation["robot_obs"][:3], dtype=float)
            xy_delta = target[:2] - current[:2]
            xy_distance = float(np.linalg.norm(xy_delta))
            orn_error = float(np.max(np.abs(orientation_error(observation))))
            if xy_distance <= threshold and orn_error <= orientation_tolerance:
                return RecoveryResult(
                    True,
                    observation,
                    steps,
                    "reached_xy_threshold",
                    target_id,
                    target,
                    xy_distance,
                    actions,
                    orn_error,
                )
            action = np.zeros(7, dtype=np.float32)
            # Do not command all the way to the target: only close the excess
            # distance outside the configured threshold.
            # Aim a few millimetres inside the boundary so float32 action
            # quantization cannot leave the TCP infinitesimally outside it.
            if xy_distance > threshold:
                travel = min(max_delta, xy_distance - stop_distance)
                action[:2] = xy_delta / xy_distance * travel / max_delta
            fill_orientation(action, observation)
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
                    float(np.max(np.abs(orientation_error(observation)))),
                )

        final_xy_distance = float(
            np.linalg.norm(
                np.asarray(observation["robot_obs"][:2], dtype=float) - target[:2]
            )
        )
        final_orientation_error = float(
            np.max(np.abs(orientation_error(observation)))
        )
        reached = (
            final_xy_distance <= threshold
            and final_orientation_error <= orientation_tolerance
        )
        return RecoveryResult(
            reached,
            observation,
            steps,
            "reached_xy_and_orientation_threshold"
            if reached
            else "motion_timeout",
            target_id,
            target,
            final_xy_distance,
            actions,
            final_orientation_error,
        )


def _wrapped_angle_delta(current, target):
    delta = np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
    return (delta + np.pi) % (2.0 * np.pi) - np.pi
