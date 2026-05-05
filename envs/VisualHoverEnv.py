from typing import Dict, Optional

import torch as th
from habitat_sim import SensorType

from .base.droneGymEnv import DroneGymEnvsBase
from ..utils.type import TensorDict


class VisualHoverEnv(DroneGymEnvsBase):
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            requires_grad: bool = False,
            random_kwargs: Optional[dict] = None,
            dynamics_kwargs: dict = {},
            scene_kwargs: dict = {},
            sensor_kwargs: list = [],
            device: str = "cpu",
            target: Optional[th.Tensor] = None,
            max_episode_steps: int = 256,
            tensor_output: bool = False,
    ):
        random_kwargs = {
            "state_generator": {
                "class": "Uniform",
                "kwargs": [
                    {"position": {"mean": [1.0, 0.0, 1.5], "half": [1.0, 1.0, 0.5]}},
                ],
            }
        } if random_kwargs is None else random_kwargs
        sensor_kwargs = [{
            "sensor_type": SensorType.DEPTH,
            "uuid": "depth",
            "resolution": [64, 64],
        }] if not sensor_kwargs else sensor_kwargs

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
            tensor_output=tensor_output,
        )

        self.target = th.ones((self.num_envs, 1)) @ th.as_tensor(
            [1.0, 0.0, 1.5] if target is None else target
        ).reshape(1, -1)
        self.success_radius = 0.5

    def get_success(self) -> th.Tensor:
        return th.full((self.num_agent,), False)

    def get_reward(self, predicted_obs=None) -> th.Tensor:
        return (
            0.1
            + (self.position - self.target).norm(dim=1) * (-0.1 / 9)
            + (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001
            + self.velocity.norm(dim=1) * -0.002
            + self.angular_velocity.norm(dim=1) * -0.002
        )

    def get_observation(
            self,
            indices=None,
            predicted_obs=None,
    ) -> Dict:
        state = th.hstack([
            (self.target - self.position) / 10,
            self.orientation,
            self.velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)

        return TensorDict({
            "state": state,
            "depth": th.as_tensor(self.sensor_obs["depth"] / 10).clamp(max=1),
        })
