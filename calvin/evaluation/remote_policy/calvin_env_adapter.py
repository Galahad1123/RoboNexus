"""CALVIN simulator construction and checkpoint-feature adaptation."""

from pathlib import Path

import numpy as np


class CalvinEnvAdapter:
    def __init__(self, env, policy_metadata):
        self.env = env
        self.metadata = policy_metadata
        self.image_keys = list(policy_metadata["image_keys"])
        expected = {
            "observation.images.static",
            "observation.images.gripper",
        }
        if set(self.image_keys) != expected:
            raise ValueError(
                "CALVIN adapter expects image keys {}, checkpoint declares {}".format(
                    sorted(expected), self.image_keys
                )
            )
        if int(policy_metadata["state_dim"]) != 15:
            raise ValueError(
                "CALVIN adapter expects a 15-D robot state, got {}".format(
                    policy_metadata["state_dim"]
                )
            )
        if int(policy_metadata["action_dim"]) != 7:
            raise ValueError(
                "CALVIN adapter expects a 7-D relative action, got {}".format(
                    policy_metadata["action_dim"]
                )
            )

    @classmethod
    def create(cls, dataset_split, policy_metadata, show_gui=False, use_egl=True):
        import hydra
        from omegaconf import OmegaConf

        dataset_split = Path(dataset_split)
        config = OmegaConf.load(dataset_split / ".hydra" / "merged_config.yaml")
        if "tactile" in config.cameras:
            del config.cameras["tactile"]
        if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            hydra.initialize(".")
        env = hydra.utils.instantiate(
            config.env,
            show_gui=bool(show_gui),
            use_egl=bool(use_egl),
            use_vr=False,
            use_scene_info=True,
        )
        return cls(env, policy_metadata)

    def reset(self, robot_obs, scene_obs):
        robot_obs = np.asarray(robot_obs, dtype=np.float64)
        observation = self.env.reset(
            robot_obs=robot_obs,
            scene_obs=np.asarray(scene_obs, dtype=np.float64),
        )
        # CALVIN Robot.reset restores joint positions and physical gripper
        # width, but not its internal last-command field. This matters for
        # held-object episodes: leaving the default +1 both corrupts state[14]
        # and can open the gripper on subsequent control.
        gripper_command = int(np.sign(robot_obs[14]))
        if gripper_command not in (-1, 1):
            raise ValueError(
                "Expected restored gripper command to be -1 or 1, got {}".format(
                    robot_obs[14]
                )
            )
        self.env.robot.gripper_action = gripper_command
        observation["robot_obs"][14] = gripper_command
        return observation

    def observation_for_policy(self, observation):
        rgb = observation["rgb_obs"]
        result = {
            "observation.state": np.asarray(observation["robot_obs"], dtype=np.float32),
            "observation.images.static": self._prepare_image(
                rgb["rgb_static"], "observation.images.static"
            ),
            "observation.images.gripper": self._prepare_image(
                rgb["rgb_gripper"], "observation.images.gripper"
            ),
        }
        return result

    def _prepare_image(self, image, key):
        import cv2

        image = np.asarray(image)
        _, target_height, target_width = self.metadata["image_shapes"][key]
        if image.shape[:2] != (target_height, target_width):
            image = cv2.resize(
                image,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )
        return np.ascontiguousarray(image, dtype=np.uint8)

    def step(self, action):
        action = self.prepare_action(action)
        self.last_executed_action = action.copy()
        return self.env.step(action)

    @staticmethod
    def prepare_action(action):
        # CALVIN mutates relative-action slices while converting them to an
        # absolute target. Never pass through a read-only transport buffer.
        action = np.array(action, dtype=np.float32, copy=True).reshape(-1)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Invalid CALVIN action shape/value: {}".format(action))
        # CALVIN relative Cartesian/rotation commands are normalized to
        # [-1, 1]. Its gripper controller is discrete and asserts exact -1/+1,
        # while a diffusion policy naturally predicts a continuous value.
        action[:6] = np.clip(action[:6], -1.0, 1.0)
        action[6] = 1.0 if action[6] >= 0.0 else -1.0
        return action

    def get_info(self):
        return self.env.get_info()

    def close(self):
        self.env.close()
        # Avoid CALVIN's destructor attempting a duplicate disconnect.
        self.env.ownsPhysicsClient = False
