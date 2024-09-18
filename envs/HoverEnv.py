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


class HoverEnv(DroneGymEnvsBase):
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
        sensor_kwargs = [{
            "sensor_type": SensorType.DEPTH,
            "uuid": "depth",
            "resolution": [64, 64],
        }]

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
            latent_dim=latent_dim,

        )

        self.target = th.ones((self.num_envs, 1)) @ th.as_tensor([9, 0., 1] if target is None else target).reshape(1,-1)
        self.success_radius = 0.5

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        if not self.requires_grad:
            obs = TensorDict({
                "state": self.state.cpu().numpy(),
            })
        else:
            obs = TensorDict({
                "state": self.state,
            })

        if self.latent is not None:
            if not self.requires_grad:
                obs["latent"] = self.latent.cpu().numpy()
            else:
                obs["latent"] = self.latent

        return obs

    def get_success(self) -> th.Tensor:
        return th.full((self.num_agent,), False)
        # return (self.position - self.target).norm(dim=1) < self.success_radius

    def get_reward(self) -> th.Tensor:
        base_r = 0.1
        pos_factor = -0.1 * 1/9
        reward = (
                base_r +
                 (self.position - self.target).norm(dim=1) * pos_factor +
                 (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 +
                 (self.velocity - 0).norm(dim=1) * -0.002 +
                 (self.angular_velocity - 0).norm(dim=1) * -0.002
        )

        return reward


class HoverEnv2(HoverEnv):

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
        sensor_kwargs = [{
            "sensor_type": SensorType.DEPTH,
            "uuid": "depth",
            "resolution": [64, 64],
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
            max_episode_steps=max_episode_steps,
            target=target,
            latent_dim=latent_dim
        )

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        state = th.hstack([
            (self.target - self.position) / self.max_sense_radius,
            self.orientation,
            self.velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)

        if not self.requires_grad:
            return TensorDict({
                "state": state.cpu().clone().numpy(),
            })

        else:
            return TensorDict({
                "state": state,
            })


