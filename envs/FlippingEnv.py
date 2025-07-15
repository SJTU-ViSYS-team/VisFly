import torch as th
from gymnasium import spaces
import os
import sys

# Add the project root to Python path to enable imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from VisFly.utils.type import TensorDict
class FlippingEnv(DroneGymEnvsBase):
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            seed: int = 42,
            visual: bool = False,
            requires_grad: bool = False,
            random_kwargs: dict = {},
            dynamics_kwargs: dict = {},
            scene_kwargs: dict = {},
            sensor_kwargs: list = [],
            device: str = "cpu",
            max_episode_steps: int = 256,
    ):
        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
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

    def get_observation(self, indices=None) -> dict:
        return TensorDict({
            "state": self.state,
        })

    def get_success(self) -> th.Tensor:
        # count half-flip crossings; require two full flips (four half flips)
        w = self.orientation[:, 0]
        current_sign = th.sign(w)
        # detect sign change as half flip
        sign_changes = current_sign != self.prev_w_sign
        # update half-flip count
        self.half_flip_counts = self.half_flip_counts + sign_changes.long()
        # update previous sign
        self.prev_w_sign = current_sign
        # compute full flips
        full_flips = self.half_flip_counts // 2
        # after two full flips, require stable orientation and low angular velocity
        stable_orientation = w > 0.9
        stable_omega = self.angular_velocity.norm(dim=1) < 0.1
        return (full_flips >= 2) & stable_orientation & stable_omega

    def get_failure(self) -> th.Tensor:
        return self.is_collision

    def get_reward(self) -> th.Tensor:
        return self.get_analytical_reward()

    def get_analytical_reward(self) -> th.Tensor:
        # 1. Roll Axes - reward for roll angular velocity around x-axis
        # roll_rate = self.angular_velocity[:, 0].clone().abs()
        # roll_reward = 0.1 * roll_rate

        # 2. Z-axis alignment reward
        current_z_axis = self._quat_rotate_vector(
            self.orientation.clone(),
            th.tensor([0., 0., 1.], device=self.device)
        )
        target_z_axis = th.tensor([0., 0., 1.], device=self.device).expand_as(current_z_axis)
        omega = 3.0
        target_z_circle = th.tensor([th.zeros_like(self.t), th.sin(self.t * omega), th.cos(self.t * omega)], device=self.device)
        z_alignment = th.sum(current_z_axis * target_z_circle, dim=1)
        # orientation_reward = 0.5 * z_alignment

        # 3. Velocity Penalty
        # linear_vel_penalty = -0.001 * self.velocity.clone().norm(dim=1)
        # angular_vel_penalty = -0.01 * self.angular_velocity[:, 1:].clone().norm(dim=1)

        # 4. Position Reward
        # target_position = th.tensor([0., 0., 1.], device=self.device).expand_as(self.position)
        # position_error = th.norm(self.position.clone() - target_position, dim=1)
        # position_reward = -0.01 * position_error

        # 5. Dense orientation reward - based on quaternion real part
        # quat_real = self.orientation[:, 0].clone()
        # dense_orientation_reward = 0.1 * quat_real

        # Combine all components
        total_reward = (
            # roll_reward
            + z_alignment
            # + linear_vel_penalty
            # + angular_vel_penalty
            # + position_reward
            # + dense_orientation_reward
        )

        # Bonus for success
        total_reward = total_reward + self._success.to(total_reward.dtype) * 10.0
        return total_reward

    def _quat_rotate_vector(self, quat: th.Tensor, vec: th.Tensor) -> th.Tensor:
        """Rotate a vector by a quaternion."""
        if vec.dim() == 1:
            vec = vec.unsqueeze(0).expand(quat.shape[:-1] + (-1,))

        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        vx, vy, vz = vec[..., 0], vec[..., 1], vec[..., 2]

        ww, xx, yy, zz = w*w, x*x, y*y, z*z
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z

        m00 = ww + xx - yy - zz
        m01 = 2 * (xy - wz)
        m02 = 2 * (xz + wy)
        m10 = 2 * (xy + wz)
        m11 = ww - xx + yy - zz
        m12 = 2 * (yz - wx)
        m20 = 2 * (xz - wy)
        m21 = 2 * (yz + wx)
        m22 = ww - xx - yy + zz

        rotated_x = m00 * vx + m01 * vy + m02 * vz
        rotated_y = m10 * vx + m11 * vy + m12 * vz
        rotated_z = m20 * vx + m21 * vy + m22 * vz

        return th.stack([rotated_x, rotated_y, rotated_z], dim=-1)

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        # initialize flip counter on reset
        w = self.orientation[:, 0]
        self.prev_w_sign = th.sign(w)
        self.half_flip_counts = th.zeros(self.num_envs, dtype=th.int64, device=self.device)
        return obs 
    
if __name__ == "__main__":
    # Make sure the quat_rotate_vector is aligned with the R function
    from VisFly.utils.maths import Quaternion
    import torch as th
    env = FlippingEnv(
        num_agent_per_scene=1,
        seed=42,
        visual=False,
        requires_grad=False,
        random_kwargs={
            "state_generator": {
                "class": "Uniform",
                "kwargs": [
                    {"position": {"mean": [0., 0., 0.], "half": [0.0, 0.0, 0.0]},
                    "orientation": {"mean": [0.5235, 0., 0.], "half": [0.0, 0.0, 0.0]}},
                ]
            }
        }
    )
    env.reset()
    print(env.orientation)
    print(Quaternion(w=env.orientation[:, 0], x=env.orientation[:, 1], y=env.orientation[:, 2], z=env.orientation[:, 3]).R)
    print(env._quat_rotate_vector(env.orientation, th.tensor([0., 0., 1.])))