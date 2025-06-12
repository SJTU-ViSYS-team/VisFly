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


def get_along_vertical_vector(base, obj):
    base_norm = base.norm(dim=1, keepdim=True)
    obj_norm = obj.norm(dim=1, keepdim=True)
    base_normal = base / (base_norm + 1e-8)
    along_obj_norm = (obj * base_normal).sum(dim=1, keepdim=True)
    along_vector = base_normal * along_obj_norm
    vertical_vector = obj - along_vector
    vertical_obj_norm = vertical_vector.norm(dim=1)
    return along_obj_norm.squeeze(), vertical_obj_norm, base_norm.squeeze()


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
            sensor_kwargs: list = {},
            device: str = "cpu",
            target: Optional[th.Tensor] = None,
            max_episode_steps: int = 256,
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
        )

        self.target = th.ones((self.num_envs, 1)) @ th.as_tensor([9, 0., 1] if target is None else target).reshape(1, -1)
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
        thrd_perce = th.pi / 18
        reward = base_r * 0 + \
                 ((self.velocity * (self.target - self.position)).sum(dim=1) / (1e-6 + (self.target - self.position).norm(dim=1))).clamp_max(10) * 0.01 + \
                 (((self.direction * self.velocity).sum(dim=1) / (1e-6 + self.velocity.norm(dim=1)) / 1).clamp(-1., 1.).acos().clamp_min(thrd_perce) - thrd_perce) * -0.01 + \
                 (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 + \
                 (self.velocity - 0).norm(dim=1) * -0.002 + \
                 (self.angular_velocity - 0).norm(dim=1) * -0.002 + \
                 1 / (self.collision_dis + 0.2) * -0.01 + \
                 (1 - self.collision_dis).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / (1e-6 + self.collision_dis)).relu() * -0.005 + \
                 self._success * (self.max_episode_steps - self._step_count) * base_r * (0.2 + 0.8 / (1 + 1 * self.velocity.norm(dim=1)))

        return reward


class NavigationEnv2(DroneGymEnvsBase):
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            requires_grad: bool = False,
            random_kwargs: dict = None,
            dynamics_kwargs: dict = None,
            scene_kwargs: dict = None,
            sensor_kwargs: list = None,
            device: str = "cpu",
            tensor_output: bool = False,
            target: Optional[th.Tensor] = None,
            max_episode_steps: int = 256,
    ):
        random_kwargs = {
            "state_generator":
                {
                    "class": "Uniform",
                    "kwargs": [
                        {"position": {"mean": [9., 0., 1.5], "half": [8.0, 6., 1.]},
                         # {"position": {"mean": [2., 0., 1.5], "half": [1.0, 6., 1.]},
                          # "orientation": {"mean": [0., 0, 0], "half": [0, 0, 180.]},
                         },
                    ]
                }
        } if random_kwargs is None else random_kwargs

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
            tensor_output=tensor_output
        )
        self.max_sense_radius = 10
        self.target = th.tile(th.as_tensor([14, 0., 1] if target is None else target), (self.num_envs, 1))
        # self.encoder = encoder
        # self.encoder.load_state_dict(th.load(os.path.dirname(__file__) + '/../utils/tools/depth_autoencoder.pth'))
        # self.encoder.eval()
        # self.encoder.requires_grad_(False)
        self.success_radius = 0.5
        self.observation_space["collision_vector"] = \
            spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def get_success(self) -> th.Tensor:
        return (self.position - self.target).norm(dim=1) <= self.success_radius
        # return th.zeros(self.num_envs, device=self.device, dtype=th.bool)

    def get_failure(self) -> th.Tensor:
        return self.is_collision

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        # scale = (self.target - self.position).norm(dim=1, keepdim=True).detach().clamp_min(self.max_sense_radius)
        # rescale_state = self.envs.transform_2_local_axes((self.target - self.position) / scale)
        rela_pos = (self.target - self.position)
        # rescale_state = rela_pos.sign() * th.log(rela_pos.abs() + 1)
        # rescale_state = self.envs.transform_2_local_axes(rescale_state)

        state = th.hstack([
            rela_pos,
            self.orientation,
            self.velocity,
            self.angular_velocity,
        ]).to(self.device)

        return TensorDict({
            "state": state,
            # "depth": th.as_tensor(self.sensor_obs["depth"] / 10).clamp(max=1),
            "collision_vector": self.collision_vector.to(self.device),
        })

    # For VisFly Manuscript
    def get_reward(self) -> th.Tensor:
        # precise and stable target flight
        base_r = 0.1
        thrd_perce = th.pi / 18

        # (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 + \
        reward = base_r * 0 + \
                 ((self.velocity * (self.target - self.position)).sum(dim=1) / (1e-6 + (self.target - self.position).norm(dim=1))).clamp_max(10) * 0.01 + \
                 (((self.direction * self.velocity).sum(dim=1) / (1e-6 + self.velocity.norm(dim=1)) / 1).clamp(-1., 1.).acos().clamp_min(thrd_perce) - thrd_perce) * -0.01 + \
                 (self.velocity - 0).norm(dim=1) * -0.002 + \
                 (self.angular_velocity - 0).norm(dim=1) * -0.01 + \
                 1 / (self.collision_dis + 0.2) * -0.01 + \
                 (1 - self.collision_dis).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / (1e-6 + self.collision_dis)).relu() * -0.01 + \
                 self._success * (self.max_episode_steps - self._step_count) * base_r * (0.2 + 0.8 / (1 + 1 * self.velocity.norm(dim=1)))

        target_approaching_v, target_away_v, target_dis = get_along_vertical_vector(self.target - self.position, self.velocity)
        obstacle_approaching_v, obstacle_away_v, collision_dis = get_along_vertical_vector(self.collision_vector, self.velocity)

        r_obstacle_spd = (obstacle_approaching_v) * -0.05 * (1 - self.collision_dis).relu()  # \
        # + obstacle_away_v * 0.01
        r_obstacle_dis = 1 / (collision_dis + 0.03) * -0.01
        require_spd = (target_dis * 2).clamp(0.5, 10)
        # target_spd_r = (1-(target_approaching_v-require_spd).abs()/require_spd) * 0.2
        target_dis_scale = 1  # / (target_dis / 3 + 0.2)
        r_target_spd = (target_approaching_v - target_away_v * target_dis_scale) * 0.02  # - target_away_v * 0.01
        r_target_dis = (self.position - self.target).norm(dim=1) * 0.01
        r_view_aware = (
                               (
                                       (self.direction * self.velocity).sum(dim=1) / (1e-6 + self.velocity.norm(dim=1))
                               ).clamp(-1., 1.).acos()
                               - thrd_perce).relu() * -0.01
        r_omega = (self.angular_velocity - 0).norm(dim=1) * -0.001
        r_collision = self.is_collision * -1
        r_success = self._success * 1
        reward = (0
                  # + r_obstacle_spd
                  + r_target_spd
                  # + r_view_aware
                  # + r_obstacle_dis
                  + r_omega
                  # + r_collision
                  + r_success)

        metrics = {
            "r_obstacle_spd": r_obstacle_spd.clone().detach(),
            "r_target_spd": r_target_spd.clone().detach(),
            "r_view_aware": r_view_aware.clone().detach(),
            "r_obstacle_dis": r_obstacle_dis.clone().detach(),
            "r_omega": r_omega.clone().detach(),
            "r_collision": r_collision.clone().detach(),
            "r_success": r_success.clone().detach()
        }
        return reward, metrics

    def get_analytical_reward(self,
                              dyn,
                              collision_vector,
                              is_collision,
                              success,
                              ) -> th.Tensor:
        base_r = 0.1
        thrd_perce = th.pi / 18
        target_approaching_v, target_away_v, target_dis = \
            get_along_vertical_vector(self.target - dyn.position, dyn.velocity)
        obstacle_approaching_v, obstacle_away_v, collision_dis = \
            get_along_vertical_vector(collision_vector, dyn.velocity)
        obstacle_spd_r = obstacle_approaching_v.squeeze() * -0.1 * (1 - collision_dis).relu()
        obstacle_dis_r = 1 / (collision_dis + 0.03) * -0.02
        require_spd = (target_dis * 2).clamp(0.5, 10)
        target_spd_r = (target_approaching_v - target_away_v) * 0.02

        view_aware_r = (
                               (
                                       (dyn.direction * dyn.velocity).sum(dim=1) / (1e-6 + dyn.velocity.norm(dim=1))
                               ).clamp(-1., 1.).acos()
                               - thrd_perce).relu() * -0.01

        reward = obstacle_spd_r \
                 + target_spd_r \
                 + view_aware_r \
                 + obstacle_dis_r \
                 + (dyn.angular_velocity - 0).norm(dim=1) * -0.01 \
                 + is_collision * -2 \
                 + success * 5

        return reward

    # def to(self, device):
    #     super().to(device)
    #     self.encoder.to(device)
