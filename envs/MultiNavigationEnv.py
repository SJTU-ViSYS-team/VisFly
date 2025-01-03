import numpy as np
from envs.multiDroneGymEnv import MultiDroneGymEnvBase
from typing import Union, Tuple, List, Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces


class MultiNavigationEnv(MultiDroneGymEnvBase):
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
            max_episode_steps: int = 256
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
        )

        self.target = th.vstack( [th.tensor([[13, -2., 1.5],
                                 [13, -0., 1.5],
                                 [13, 2., 1.5]]) for i in range(num_scene)] )\
            if target is None else target

        self.observation_space["target"] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.observation_space["swarm"] = spaces.Box(low=-np.inf, high=np.inf,
                                                     shape=(self.num_agent_per_scene-1, self.observation_space["state"].shape[0]),
                                                     dtype=np.float32)
        # self.observation_space["swarm"].shape = (self.num_agent_per_scene-1, self.observation_space["swarm"].shape[0])
        self.success_radius = 0.5
        self.is_find_path = scene_kwargs.get("is_find_path", False)
        if self.is_find_path:
            self._paths = [None for _ in range(self.num_envs)]

    def reset_by_id(self, indices):
        obs = super().reset_agent_by_id(indices)
        if self.is_find_path:
            for indice in indices:
                self._paths[indice] = self.envs.find_path(self.target[indice])
        return obs

    def reset(self):
        obs = super().reset()
        if self.is_find_path:
            self._paths = self.envs._find_paths(target=self.target)
        return obs

    @property
    def path(self):
        return self._paths

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        swarm = th.zeros((self.num_agent, self.num_agent_per_scene-1, self.observation_space["state"].shape[0]))
        for scene_index in range(self.num_scene):
            for agent_index in range(self.num_agent_per_scene):
                agent_i = scene_index * self.num_agent_per_scene + agent_index
                sequence = th.cat([th.arange(scene_index*self.num_agent_per_scene, agent_i, dtype=th.uint8),
                                   th.arange(agent_i+1,(scene_index+1)*self.num_agent_per_scene, dtype=th.uint8)])
                swarm[agent_i, :, :] = self.state[sequence.tolist(),:]

        if not self.requires_grad:
            return {
                "state": self.state.cpu().clone().numpy(),
                "depth": self.sensor_obs["depth"],
                "target": self.target.cpu().clone().numpy(),
                "swarm": swarm.cpu().clone().numpy(),
            }
        else:
            return {
                "state": self.state.to(self.device),
                "depth": self.sensor_obs["depth"],
                "target": self.target.to(self.device),
                "swarm": swarm.to(self.device)
            }

    def get_done(self) -> th.Tensor:
        done = th.full((self.num_envs, ), False)
        # for i in range(self.num_scene):
        #     done[i*self.num_agent_per_scene:(i+1)*self.num_agent_per_scene] = self.is_collision[i*self.num_agent_per_scene:(i+1)*self.num_agent_per_scene].all()
        return done

    def get_success(self) -> th.Tensor:
        success = self.position[:,0] > 10
        return success
        # return (self.position - self.target).norm(dim=1) <= self.success_radius)

        # (self.position - self.target).norm(dim=1) <= self.success_radius
        # return th.full((self.num_agent,), False).to(self.device)

    def get_reward(self) -> th.Tensor:
        # precise and stable target flight
        base_r = 0.1
        """reward function"""
        ref_factor = 20
        # reward = base_r + \
        #     th.tanh((self.position-self.target).norm(dim=1) * -0.01 * ref_factor) / ref_factor + \
        #     (self.orientation - th.tensor([1, 0, 0, 0]).to(self.device)).norm(dim=1) * -0.00001  + \
        #     (self.velocity-0).norm(dim=1) * -0.002 + \
        #     (self.angular_velocity-0).norm(dim=1) * -0.002 + \
        #     self._success * (self.max_episode_steps - self._step_count) * base_r * 2 / ((self.velocity-0).norm(dim=1)+1)
        # self.closest_obstacle_dis * 0.01 + \

        """no ultimate stability reward function"""
        thrd_perce = th.pi/18
        sigma = 0.1
        # (self.velocity - 0).norm(dim=1) * -0.002 / (th.max((self.position - self.target).norm(dim=1) - self.success_radius + sigma, th.as_tensor(sigma)) ** 2) + \
        thrd_perce = th.pi/18
        reward = base_r*0 + \
                ((self.velocity *(self.target - self.position)).sum(dim=1) / (1e-6+(self.target - self.position).norm(dim=1))).clamp_max(10) * 0.01+\
                 (((self.direction * self.velocity).sum(dim=1) / (1e-6+self.velocity.norm(dim=1)) / 1).clamp(-1., 1.).acos().clamp_min(thrd_perce)-thrd_perce)*-0.01+\
                 (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 + \
                 (self.velocity - 0).norm(dim=1) * -0.002 + \
                 (self.angular_velocity - 0).norm(dim=1) * -0.002 + \
                 1 / (self.collision_dis + 0.2) * -0.01 + \
                 (1-self.collision_dis ).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / (1e-6+self.collision_dis)).relu() * -0.005 + \
                 self._success * (self.max_episode_steps - self._step_count) * base_r * (0.5+0.5/ (1+1*self.velocity.norm(dim=1)))

        return reward
        # (self.collision_dis <= 1) * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / self.collision_dis) * -0.02 +


# (1 - self.collision_dis).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / self.collision_dis).relu() * -0.01 + \
