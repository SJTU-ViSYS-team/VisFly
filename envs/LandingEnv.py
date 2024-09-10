import numpy as np
from envs.droneGymEnv import DroneGymEnvsBase
from typing import Union, Tuple, List, Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt


class LandingEnv(DroneGymEnvsBase):
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            requires_grad: bool = False,
            random_kwargs: dict = {},
            dynamics_kwargs: dict = {},
            scene_kwargs: dict = {},
            sensor_kwargs: list = [],
            device: str = "cpu",
            target: Optional[th.Tensor] = None,
            max_episode_steps: int = 128
    ):
        sensor_kwargs = [{
            "sensor_type": SensorType.COLOR,
            "uuid": "color",
            "resolution": [64, 64],
            "orientation": [-np.pi / 2, 0, 0]
        }]

        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
            device=device,
            max_episode_steps=max_episode_steps
        )

        self.target = th.tensor([2, 0, 0])
        self.success_radius = 0.5
        self.observation_space["target"] = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.centers = None

    def reset_by_id(self, indices):
        obs = super().reset_by_id(indices)
        return obs

    def reset(self):
        obs = super().reset()
        return obs

    def get_done(self):
        out_of_vision = th.as_tensor(self.centers).isnan().any(dim=1)
        return out_of_vision

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        two_value = (self.sensor_obs["color"].mean(axis=1) < 70)
        self._pre_centers = self.centers if self.centers is not None else None
        self.centers = th.as_tensor([center_of_mass(each_img) for each_img in two_value]) \
                       / self.observation_space["color"].shape[1] - 0.5
        out_of_vision = th.as_tensor(self.centers).isnan().any(dim=1)
        for i in th.arange(self.num_envs)[out_of_vision]:
            self.centers[i] = self._pre_centers[i]
        # debug
        # plt.imshow(self.sensor_obs["color"][0])
        # plt.show()
        # import cv2 as cv
        # cv.imshow("2 value", np.full_like(two_value[0], 255, dtype=np.uint8) * two_value[0])
        # cv.imshow("color", self.sensor_obs["color"][0])
        # cv.waitKey(0)
        if not self.requires_grad:
            return {
                "state": self.state.cpu().clone().numpy(),
                "color": self.sensor_obs["color"],
                "target": self.centers,
            }

        else:
            return {
                "state": self.state.to(self.device),
                "color": th.from_numpy(np.stack([np.expand_dims(agent["depth"], 0) for agent in self.sensor_obs])).to(self.device),
            }

    def get_success(self) -> th.Tensor:
        landing_half = 0.25
        # return th.full((self.num_envs,), False)
        return (self.position[:, 2] <= 0.15) \
            & ((self.position[:, :2] < (self.target[:2] + landing_half)).all(dim=1) & (self.position[:, :2] > (self.target[:2] - landing_half)).all(dim=1))
        # & (self.velocity.norm(dim=1) <= 0.1)) #
        # & \
        # ((self.position[:, :2] < self.target[:2] + landing_half).all(dim=1) & (self.position[:, :2] > self.target[:2] - landing_half).all(dim=1))

        # & (self.velocity.norm(dim=1) <= 0.1) \

    def get_reward(self) -> th.Tensor:
        # precise and stable target flight
        base_r = 0.1
        """reward function"""
        # 0.1 * (1 - self.centers.norm(dim=1) / 0.707)
        # reward = 0.1 * (3 - (self.position[:,:2]-self.target[:2]).norm(dim=1)).relu() / 3 + \
        #          (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.0001 + \
        #          0.1 * (3- self.position[:, 2]).relu() / 3 + \
        #          -0.01 * self.velocity.norm(dim=1) + \
        #          (self.angular_velocity - 0).norm(dim=1) * -0.002 # + \
        # self._success * (self.max_episode_steps - self._step_count) * base_r * 1.8 / (self.velocity.norm(dim=1) + 1)
        # + \
        # self._success * (1 - self.centers.norm(dim=1) / 0.707) * base_r * 4 + \
        # self._success * (self.max_episode_steps - self._step_count) * base_r * 4 / (self.velocity.norm(dim=1) + 1)  # + \

        # self._success * (self.max_episode_steps - self._step_count) * (1 - self.centers.norm(dim=1) / 0.707) * base_r * 8

        # 0.02 * (- self.velocity[:, 2]) + \
        # ((self.position[:, 2] <= 0.15) & (self.velocity.norm(dim=1) <= 0.1)) * (self.max_episode_steps - self._step_count) * base_r * 4 + \

        reward = 0.2 * (1.25 - self.centers.norm(dim=1) / 1).clamp_max(1.) + \
                 (self.orientation[:, [0, 1]]).norm(dim=1) * -0.2 + \
                 0.1 * (3 - self.position[:, 2]).clamp(0, 3) / 3 * 2 + \
                 -0.02 * self.velocity.norm(dim=1) + \
                 -0.01 * self.angular_velocity.norm(dim=1) + \
                 0.1 * 20 * self._success * (10 + (self.max_episode_steps - self._step_count)) / (1+2*self.velocity.norm(dim=1)) # / (self.velocity.norm(dim=1) + 1)

        # succeed
        # reward = 0.1 * (3 - (self.position-self.target).norm(dim=1)).relu() / 3 + \
        #          (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.0001 + \
        #          -0.002 * self.velocity.norm(dim=1) + \
        #          -0.002 * self.angular_velocity.norm(dim=1) + \
        #          0.1 * 2 * self._success * (self.max_episode_steps - self._step_count)

        return reward

# self.is_collision * -0.5 +
