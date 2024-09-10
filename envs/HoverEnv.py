import os
import sys

import numpy as np
from envs.droneGymEnv import DroneGymEnvsBase
from typing import Union, Tuple, List, Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces
from examples.cluttered_flight_mb.train_encoder import model as encoder
from utils.type import TensorDict


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

        self.target = th.ones((self.num_envs, 1)) @ th.tensor([[9, 0., 1]]) if target is None else target
        # self.target = th.ones((self.num_envs, 1)) @ th.tensor([[1, 0., 1]]) if target is None else target

        self.observation_space["target"] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.success_radius = 0.5
        self.is_find_path = scene_kwargs.get("is_find_path", False)
        if self.is_find_path:
            self._paths = [None for _ in range(self.num_envs)]

    def reset_by_id(self, indices):
        obs = super().reset_by_id(indices)
        if self.is_find_path:
            for indice in indices:
                self._paths[indice] = self.envs.find_path(self.target[indice])
        return obs

    def reset(self):
        obs = super().reset()
        return obs

    @property
    def path(self):
        return self._paths

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        if not self.requires_grad:
            if self.visual:
                return {
                    "state": self.state.cpu().clone().numpy(),
                    "depth": self.sensor_obs["depth"],
                    "target": self.target.cpu().clone().numpy(),
                    # "latent": self.latent.cpu().clone(),
                }
            else:
                return {
                    "state": self.state.cpu().clone().numpy(),
                    "target": self.target.cpu().clone().numpy()
                }
        else:
            if self.visual:
                return {
                    "state": self.state.to(self.device),
                    "depth": th.from_numpy(self.sensor_obs["depth"]).to(self.device),
                    "target": self.target.to(self.device)
                }
            else:
                return {
                    "state": self.state.to(self.device),
                    "target": self.target.to(self.device)
                }

    def get_success(self) -> th.Tensor:
        return th.full((self.num_agent,), False)

    def get_reward(self) -> th.Tensor:
        base_r = 0.1
        reward = (
                base_r +
                (self.position - self.target).norm(dim=1) * -0.01 +
                (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 +
                (self.velocity - 0).norm(dim=1) * -0.002 +
                (self.angular_velocity - 0).norm(dim=1) * -0.002
                + self._success * (self.max_episode_steps - self._step_count) * base_r  # / ((self.velocity-0).norm()+1)
            # + (1 / self.collision_dis + 0.1) * -0.001
            # + (1 - self.collision_dis).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / (1e-6 + self.collision_dis)).relu() * -0.005
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

        self.encoder = encoder
        self.encoder.load_state_dict(th.load(os.path.dirname(os.path.abspath(sys.argv[0])) + '/saved/depth_autoencoder.pth'))
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
            "target": self.observation_space["target"],
        })

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        if not self.requires_grad:
            return TensorDict({
                "state": self.state.cpu().clone().numpy(),
                # "depth": encoder.encode(th.from_numpy(self.sensor_obs["depth"]).to(encoder.encoder[0].weight.device)).cpu().numpy(),
                "target": self.target.cpu().clone().numpy(),
                # "latent": self.latent.cpu().clone(),
            })

        else:
            return TensorDict({
                "state": self.state.to(self.device),
                # "depth": encoder.encode(th.from_numpy(self.sensor_obs["depth"]).to(self.device)).detach(),
                "target": self.target.to(self.device),
                # "depth_unencode": th.as_tensor(self.sensor_obs["depth"]),
            })

    # (1 - self.collision_dis).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / self.collision_dis).relu() * -0.01 + \
    def to(self, device):
        super().to(device)
        self.encoder.to(device)
