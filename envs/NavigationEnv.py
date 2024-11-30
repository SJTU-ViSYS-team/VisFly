import os

import numpy as np
from habitat_sim.sensor import SensorType

from .base.droneGymEnv import DroneGymEnvsBase
from typing import Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces

from ..utils.tools.train_encoder import model as encoder
from ..utils.type import TensorDict


class NavigationEnv(DroneGymEnvsBase):
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
            latent_dim=latent_dim,
        )

        self.target = th.ones((self.num_envs, 1)) @ th.as_tensor([9, 0., 1] if target is None else target).reshape(1,-1)
        self.observation_space["target"] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.success_radius = 0.5

    def get_observation(
            self,
            indices=None
    ) -> Dict:

        if self.visual:
            return TensorDict({
                "state": self.state.to(self.device),
                "depth": th.from_numpy(self.sensor_obs["depth"]).to(self.device),
                "target": self.target.to(self.device)
            })
        else:
            return TensorDict({
                "state": self.state.to(self.device),
                "target": self.target.to(self.device)
            })

    def get_success(self) -> th.Tensor:
        return (self.position - self.target).norm(dim=1) <= self.success_radius

    # For VisFly Manuscript
    def get_reward(self) -> th.Tensor:
        # precise and stable target flight
        base_r = 0.1
        thrd_perce = th.pi/18
        reward = base_r*0 + \
                ((self.velocity *(self.target - self.position)).sum(dim=1) / (1e-6+(self.target - self.position).norm(dim=1))).clamp_max(10) * 0.01+\
                 (((self.direction * self.velocity).sum(dim=1) / (1e-6+self.velocity.norm(dim=1)) / 1).clamp(-1., 1.).acos().clamp_min(thrd_perce)-thrd_perce)*-0.01+\
                 (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 + \
                 (self.velocity - 0).norm(dim=1) * -0.002 + \
                 (self.angular_velocity - 0).norm(dim=1) * -0.002 + \
                 1 / (self.collision_dis + 0.2) * -0.01 + \
                 (1-self.collision_dis ).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / (1e-6+self.collision_dis)).relu() * -0.005 + \
                 self._success * (self.max_episode_steps - self._step_count) * base_r * (0.2+0.8/ (1+1*self.velocity.norm(dim=1)))

        return reward


class NavigationEnv2(NavigationEnv):
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
        self.max_sense_radius = 10
        self.encoder = encoder
        self.encoder.load_state_dict(th.load(os.path.dirname(__file__)+'/../utils/tools/depth_autoencoder.pth'))
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self.observation_space["depth_state"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(256,),
            dtype=np.float32,
        )
        # reset observation space
        self.observation_space = spaces.Dict({
            "state": self.observation_space["state"],
            # "depth": self.observation_space["depth_state"],
            # "target": self.observation_space["target"],
        })

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

        # return TensorDict({
        #     "state": self.state.to(self.device) ,
        #     # "depth": encoder.encode(th.from_numpy(self.sensor_obs["depth"]).to(self.device)).detach(),
        #     "target": self.target.to(self.device),
        #     # "depth_unencode": th.as_tensor(self.sensor_obs["depth"]),
        # })
        return TensorDict({
            "state": state,
            # "depth": encoder.encode(th.from_numpy(self.sensor_obs["depth"]).to(self.device)),
            # "target": self.target.to(self.device),
            # "depth_unencode": th.as_tensor(self.sensor_obs["depth"]),
        })

    def to(self, device):
        super().to(device)
        self.encoder.to(device)
