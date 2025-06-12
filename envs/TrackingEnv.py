import os
import sys

import numpy as np
from .base.droneGymEnv import DroneGymEnvsBase
from typing import Union, Tuple, List, Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces
from ..utils.tools.train_encoder import model as encoder
from ..utils.type import TensorDict


class TrackEnv(DroneGymEnvsBase):
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
            max_episode_steps: int = 256,
            latent_dim=None,
    ):
        self.center = th.as_tensor([2, 0, 1])
        self.next_points_num = 10
        self.radius = 2
        self.dt = 0.1
        self.radius_spd = 0.2 * th.pi / 1
        self.success_radius = 0.5

        random_kwargs = {
            "state_generator":
                {
                    "class": "Uniform",
                    "kwargs": [
                        {"position": {"mean": [self.center[0] + self.radius, 0., self.center[2]],
                                      "half": [.2, .2, 0.2]}},
                    ]
                }
        }

        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            sensor_kwargs=sensor_kwargs,
            scene_kwargs=scene_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
        )

        self.observation_space["state"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3 * (self.next_points_num - 1) + self.observation_space["state"].shape[0],),
            dtype=np.float32
        )

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        ts = self.t.repeat(self.next_points_num, 1).T + th.arange(self.next_points_num) * self.dt
        self.target = (th.stack([self.radius * th.cos(self.radius_spd * ts) + self.center[0],
                                 self.radius * th.sin(self.radius_spd * ts) + self.center[1],
                                 th.zeros(ts.shape) + self.center[2]])
                       ).permute(1, 2, 0)
        # self.target = self.trajs[self.current_traj_index, target_index]
        diff_pos = self.target - self.position.unsqueeze(1)
        # consider target as next serveral waypoint
        diff_pos_flatten = diff_pos.reshape(self.num_envs, -1)

        state = th.hstack([
            diff_pos_flatten / self.max_sense_radius,
            self.orientation,
            self.velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)

        return TensorDict({
            "state": state,
        })

    def get_success(self) -> th.Tensor:
        return th.full((self.num_agent,), False)
        # return (self.position - self.target).norm(dim=1) < self.success_radius

    def get_reward(self) -> th.Tensor:
        base_r = 0.1
        pos_factor = -0.1 * 1 / 9
        reward = (
                base_r +
                (self.position - self.target[:, 0, :]).norm(dim=1) * pos_factor +
                (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 +
                (self.velocity - 0).norm(dim=1) * -0.002 +
                (self.angular_velocity - 0).norm(dim=1) * -0.002
        )

        return reward


class TrackEnv2(TrackEnv):

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
            max_episode_steps: int = 256,
            latent_dim=None,
    ):
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
            max_episode_steps=max_episode_steps,
            target=target,
            latent_dim=latent_dim
        )
