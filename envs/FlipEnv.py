import torch as th
from typing import Dict

from .base.droneGymEnv import DroneGymEnvsBase
from ..utils.type import TensorDict


class FlipEnv(DroneGymEnvsBase):
    """Environment for a 360-degree flip followed by stabilization."""

    def __init__(
        self,
        num_agent_per_scene: int = 1,
        num_scene: int = 1,
        seed: int = 42,
        visual: bool = True,
        requires_grad: bool = False,
        random_kwargs: dict = None,
        dynamics_kwargs: dict = None,
        scene_kwargs: dict = {},
        sensor_kwargs: list = [],
        device: str = "cpu",
        max_episode_steps: int = 256,
    ):
        random_kwargs = {
            "state_generator": {
                "class": "Uniform",
                "kwargs": [
                    {"position": {"mean": [0.0, 0.0, 1.5], "half": [0.5, 0.5, 0.2]}},
                ],
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
        )

        self.rotation_required = 2 * th.pi
        self.rotation_progress = th.zeros((self.num_envs,), device=self.device)
        self.flip_complete = th.zeros((self.num_envs,), dtype=th.bool, device=self.device)
        self.stable_steps = th.zeros((self.num_envs,), dtype=th.int32, device=self.device)
        self.required_stable_steps = 10

    def reset(self, state=None, obs=None):
        obs = super().reset(state, obs)
        self.rotation_progress[:] = 0
        self.flip_complete[:] = False
        self.stable_steps[:] = 0
        return obs

    def get_observation(self, indices=None) -> Dict:
        return TensorDict({"state": self.state})

    def get_success(self) -> th.Tensor:
        stable_ori = (self.orientation - th.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)).norm(dim=1) < 0.1
        stable_ang = self.angular_velocity.norm(dim=1) < 0.1
        stabilized = stable_ori & stable_ang
        self.stable_steps = th.where(stabilized, self.stable_steps + 1, th.zeros_like(self.stable_steps))
        return self.flip_complete & (self.stable_steps >= self.required_stable_steps)

    def get_failure(self) -> th.Tensor:
        return self.is_collision

    def get_reward(self) -> th.Tensor:
        dt = self.envs.dynamics.ctrl_dt
        self.rotation_progress = self.rotation_progress + self.angular_velocity[:, 0].abs() * dt
        self.flip_complete = self.flip_complete | (self.rotation_progress >= self.rotation_required)

        progress_error = (self.rotation_required - self.rotation_progress).clamp_min(0)
        orientation_error = (self.orientation - th.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)).norm(dim=1)
        reward = th.where(
            ~self.flip_complete,
            -progress_error * 0.05,
            -orientation_error - self.angular_velocity.norm(dim=1) * 0.5,
        )
        return reward
