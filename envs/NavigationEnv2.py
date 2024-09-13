import numpy as np
from .NavigationEnv import NavigationEnv
from typing import Optional, Dict
import torch as th
from habitat_sim import SensorType
import os, sys
from gymnasium import spaces
from ..utils.tools.train_encoder import model as encoder
from ..utils.type import TensorDict


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
        # a = th.as_tensor([-1,-1,-1,0,0,0,0,0,0,0,0,0,0]).to(self.device)
        # b = th.cat([self.target[0], 0,0,0,0,0,0,0,0,0,0]).to(self.device)
        # c = th.as_tensor([]).to(self.device)
        state = th.hstack([
            (self.target - self.position) / self.max_sense_radius,
            self.orientation,
            self.velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)
        if not self.requires_grad:
            return TensorDict({
                "state": self.state.cpu().clone().numpy(),
                # "depth": encoder.encode(th.from_numpy(self.sensor_obs["depth"]).to(self.device)).cpu().numpy(),
                # "target": self.target.cpu().clone().numpy(),
                # "latent": self.latent.cpu().clone(),
            })

        else:
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
    # (1 - self.collision_dis).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / self.collision_dis).relu() * -0.01 + \

    def to(self, device):
        super().to(device)
        self.encoder.to(device)
