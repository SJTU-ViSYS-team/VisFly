import numpy as np
from envs.droneGymEnv import DroneGymEnvsBase
from typing import Union, Tuple, List, Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces


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

        # self.target = th.ones((self.num_envs, 1)) @ th.tensor([[9, 0., 1]]) if target is None else target
        self.target = th.ones((self.num_envs, 1)) @ th.tensor([[9, 0., 1]]) if target is None else target

        self.observation_space["target"] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.success_radius = 0.5
        self.is_find_path = scene_kwargs.get("is_find_path", False)
        if self.is_find_path:
            self._paths = [None for _ in range(self.num_envs)]

    def reset_by_id(self, indices=None):
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
        return (self.position - self.target).norm(dim=1) <= self.success_radius
        # return self.position[:, 0] >= 10.
        # return th.full((self.num_agent,), False)

    # For VisFly Manuscript
    # def get_reward(self) -> th.Tensor:
    #     # precise and stable target flight
    #     base_r = 0.1
    #     thrd_perce = th.pi/18
    #     reward = base_r*0 + \
    #             ((self.velocity *(self.target - self.position)).sum(dim=1) / (1e-6+(self.target - self.position).norm(dim=1))).clamp_max(10) * 0.01+\
    #              (((self.direction * self.velocity).sum(dim=1) / (1e-6+self.velocity.norm(dim=1)) / 1).clamp(-1., 1.).acos().clamp_min(thrd_perce)-thrd_perce)*-0.01+\
    #              (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 + \
    #              (self.velocity - 0).norm(dim=1) * -0.002 + \
    #              (self.angular_velocity - 0).norm(dim=1) * -0.002 + \
    #              1 / (self.collision_dis + 0.2) * -0.01 + \
    #              (1-self.collision_dis ).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / (1e-6+self.collision_dis)).relu() * -0.005 + \
    #              self._success * (self.max_episode_steps - self._step_count) * base_r * (0.2+0.8/ (1+1*self.velocity.norm(dim=1)))
    #
    #     return reward

    def get_reward(self) -> th.Tensor:
        base_r = 0.1
        pos_factor = -0.1 * 1/9
        reward = (
                base_r +
                 (self.position - self.target).norm(dim=1) * pos_factor +
                 (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 +
                 (self.velocity - 0).norm(dim=1) * -0.002 +
                 (self.angular_velocity - 0).norm(dim=1) * -0.002
                 + self._success * (self.max_episode_steps - self._step_count) * base_r # / ((self.velocity-0).norm()+1)
                 # + (1 / self.collision_dis + 0.1) * -0.001
                 # + (1 - self.collision_dis).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / (1e-6 + self.collision_dis)).relu() * -0.005
        )
        # target_vector = self.target - self.position
        # approching_vel = ((self.velocity-0) * target_vector).sum(dim=1)/ (1e-6 + target_vector.norm(dim=1))
        # approching_vel = approching_vel.clamp_max(3)
        # approching_vel_k = (3 / (0.1 + (self.target - self.position).norm(dim=1)))
        # reward = (
        #     0.1
        #     # + approching_vel * 0.1
        #     # + self.success * 5
        #     + (self.position - self.target).norm(dim=1) * -0.01
        #
        # )
        return reward

#     def get_reward(self) -> th.Tensor:
#
#         factors = th.tensor([0.1,
#                              0.1,
#                              1,
#                              1
#                              ])
#
#         factors = factors / factors.norm()
#
#         t_perception = th.pi / 18
#         ori_r = (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.0001
#         ang_vel_r = (self.angular_velocity - 0).norm(dim=1) * -0.01
#         col_r = (2 / self.collision_dis + 0.2) * -0.01
#         target_toward_r = (((self.velocity-0) * (self.target - self.position)).sum(dim=1)/ (1e-6 + (self.target - self.position).norm(dim=1))).clamp_max(10) * 0.1
#         col_toward_r =(1-self.collision_dis ).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / (1e-6 + self.collision_dis)).relu() * -0.05
#         reward = ori_r + ang_vel_r + col_r + target_toward_r + col_toward_r
#                 # ((self.velocity * (self.target - self.position)).sum(dim=1) / (1e-6 + (self.target - self.position).norm(dim=1))).clamp_max(10) * 0.01
#                 # (((self.direction * self.velocity).sum(dim=1) / (1e-6 + self.velocity.norm(dim=1)) / 1).clamp(-1., 1.).acos().clamp_min(t_perception) - t_perception) * -0.01+
#         reward = target_toward_r
# # +  (self.target - self.position).norm(dim=1).clamp_max(10) * 0.01
#         # (1-self.collision_dis ).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / (1e-6+self.collision_dis)).relu() * -0.005 #+ \
#         # ((self.velocity * (self.target - sm(dim=1) / (1e-6 + (self.target - self.position).norm(dim=1))).clamp_max(10) * 0.01 + \
#         # (((self.direction * self.velocity).sum(dim=1) / (1e-6+self.velocity.norm(dim=1)) / 1).clamp(-1., 1.).acos().clamp_min(thrd_perce)-thrd_perce)*-0.01+\
#         # self._success * (self.max_episode_steps - self._step_count) * base_r * (0.2+0.8/ (1+1*self.velocity.norm(dim=1)))
#elf.position)).su
#         return reward
#         # (self.collision_dis <= 1) * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / self.collision_dis) * -0.02 +
#
# # (1 - self.collision_dis).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / self.collision_dis).relu() * -0.01 +
