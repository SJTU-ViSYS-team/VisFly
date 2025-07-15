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
    """
    Decompose obj vector into components along and perpendicular to base vector
    Simplified for BPTT compatibility - inputs should already be properly detached
    """
    # Safe norm computation with minimum clipping to avoid zero gradients
    base_norm = base.norm(dim=1, keepdim=True).clamp(min=1e-8)
    obj_norm = obj.norm(dim=1, keepdim=True).clamp(min=1e-8)
    
    # Safe division for normalization
    base_normal = base / base_norm
    along_obj_norm = (obj * base_normal).sum(dim=1, keepdim=True)
    along_vector = base_normal * along_obj_norm
    vertical_vector = obj - along_vector
    vertical_obj_norm = vertical_vector.norm(dim=1).clamp(min=1e-8)
    
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
            sensor_kwargs: list = [],
            device: str = "cpu",
            target: Optional[th.Tensor] = None,
            max_episode_steps: int = 256,
    ):
        # Hard-code sensor_kwargs like old_VisFly to ensure depth sensor is always available
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
            tensor_output=False,
        )

        self.target = th.ones((self.num_envs, 1)) @ th.as_tensor([9, 0., 1] if target is None else target).reshape(1, -1)
        self.observation_space["target"] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.success_radius = 0.5

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        # Match old_VisFly behavior exactly - return numpy arrays when requires_grad=False
        if not self.requires_grad:
            if self.visual:
                # Normalize depth to [0,1]
                depth_np = self.sensor_obs["depth"] / self.max_sense_radius
                depth_np = np.clip(depth_np, 0.0, 1.0)
                return TensorDict({
                    "state": self.state.cpu().clone().numpy(),
                    "depth": depth_np,
                    "target": self.target.cpu().clone().numpy(),
                })
            else:
                return TensorDict({
                    "state": self.state.cpu().clone().numpy(),
                    "target": self.target.cpu().clone().numpy(),
                })
        else:
            if self.visual:
                # Normalize depth to [0,1]
                depth_t = th.from_numpy(self.sensor_obs["depth"]).to(self.device)
                depth_t = (depth_t / self.max_sense_radius).clamp(0, 1)
                return TensorDict({
                    "state": self.state.to(self.device),
                    "depth": depth_t,
                    "target": self.target.to(self.device),
                })
            else:
                return TensorDict({
                    "state": self.state.to(self.device),
                    "target": self.target.to(self.device),
                })

    def get_success(self) -> th.Tensor:
        """Define success as reaching within self.success_radius of target."""
        # Ensure tensors are on the same device
        position = self.position.to(self.device)
        target = self.target.to(self.device)
        return (position - target).norm(dim=1) <= self.success_radius

    # For VisFly Manuscript
    def get_reward(self) -> th.Tensor:
        # precise and stable target flight
        base_r = 0.1
        thrd_perce = th.pi/18
        
        # Fix device mismatch: ensure reference quaternion is on same device as orientation
        ref_orientation = th.tensor([1, 0, 0, 0], device=self.orientation.device, dtype=self.orientation.dtype)
        
        reward = base_r*0 + \
                ((self.velocity *(self.target - self.position)).sum(dim=1) / (1e-6+(self.target - self.position).norm(dim=1))).clamp_max(10) * 0.01+\
                 (((self.direction * self.velocity).sum(dim=1) / (1e-6+self.velocity.norm(dim=1)) / 1).clamp(-1., 1.).acos().clamp_min(thrd_perce)-thrd_perce)*-0.01+\
                 (self.orientation - ref_orientation).norm(dim=1) * -0.00001 + \
                 (self.velocity - 0).norm(dim=1) * -0.002 + \
                 (self.angular_velocity - 0).norm(dim=1) * -0.002 + \
                 1 / (self.collision_dis + 0.2) * -0.01 + \
                 (1-self.collision_dis ).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / (1e-6+self.collision_dis)).relu() * -0.005 + \
                 self._success * (self.max_episode_steps - self._step_count) * base_r * (0.2+0.8/ (1+1*self.velocity.norm(dim=1)))

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

        # domain randomization: add drag noise in random_kwargs
        if "drag_random" not in random_kwargs:
            random_kwargs["drag_random"] = 0.1

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
            tensor_output=False
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

        # Add target to observation space for learning algorithms that expect it (e.g. StateTargetImageExtractor)
        self.observation_space["target"] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # When visual is enabled, include depth image channel in observation space
        if visual:
            self.observation_space["depth"] = spaces.Box(low=0.0, high=1.0, shape=(1, 64, 64), dtype=np.float32)

    def get_success(self) -> th.Tensor:
        return (self.position - self.target).norm(dim=1) <= self.success_radius
        # return th.zeros(self.num_envs, device=self.device, dtype=th.bool)

    def get_failure(self) -> th.Tensor:
        return self.is_collision

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        # compute relative position to target (global frame)
        rela_pos = (self.target - self.position).to(self.device)
        # construct proprioceptive state: relative position, orientation, velocity, angular velocity
        # Ensure all tensors are on the same device before stacking
        orient = self.orientation.to(self.device)
        vel = self.velocity.to(self.device)
        ang_vel = self.angular_velocity.to(self.device)

        state = th.hstack([
            rela_pos,
            orient,
            vel,
            ang_vel,
        ])
        # get or create depth observation and normalize
        if self.visual and "depth" in self.sensor_obs:
            depth = th.from_numpy(self.sensor_obs["depth"]).to(self.device)
        else:
            # create dummy depth when visual disabled
            depth = th.zeros((self.num_envs, 1, 64, 64), device=self.device)
        depth = (depth / self.max_sense_radius).clamp(0, 1)
        # encode depth to latent representation
        # depth_state = self.encoder.encode(depth)
        obs_dict = {
            "state": state,
            "collision_vector": self.collision_vector.to(self.device),
        }
        # Provide depth if available (or zeros if not visual)
        if self.visual:
            obs_dict["depth"] = depth
        # Expose absolute target position for the extractor (needed by StateTargetImageExtractor)
        obs_dict["target"] = self.target.to(self.device)

        return TensorDict(obs_dict)

    # For VisFly Manuscript
    def get_reward(self) -> th.Tensor:
        # precise and stable target flight
        base_r = 0.1
        thrd_perce = th.pi / 18

        # Fix device mismatch: ensure reference quaternion is on same device as orientation
        ref_orientation = th.tensor([1, 0, 0, 0], device=self.orientation.device, dtype=self.orientation.dtype)

        reward = base_r * 0 \
            + ((self.velocity * (self.target - self.position)).sum(dim=1) / (1e-6 + (self.target - self.position).norm(dim=1))).clamp_max(10) * 0.01 \
            + (((self.direction * self.velocity).sum(dim=1) / (1e-6 + self.velocity.norm(dim=1)) / 1).clamp(-1., 1.).acos().clamp_min(thrd_perce) - thrd_perce) * -0.01 \
            + (self.orientation - ref_orientation).norm(dim=1) * -0.00001 \
            + (self.velocity - 0).norm(dim=1) * -0.002 \
            + (self.angular_velocity - 0).norm(dim=1) * -0.002 \
            + 1 / (self.collision_dis + 0.2) * -0.01 \
            + (1 - self.collision_dis).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / (1e-6 + self.collision_dis)).relu() * -0.005 \
            + self._success * (self.max_episode_steps - self._step_count) * base_r * (0.2 + 0.8 / (1 + self.velocity.norm(dim=1)))

        return reward

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

class NavigationEnv3(NavigationEnv):
    """
    This class is intended to build a complete environment for depth-based end2end obstacle avoidance task.
    To make sure Sim2Real transfer, we need to add domain randomization to the environment, as well as using SysID configed drone dynamics.
    Multiple agent environments can avoid overfitting to one scene.
    We need to consider the termination setting.
    If we use velocity control, it would be too aggressive, but if we use position control, it would be too slow.
    Yaw randomization so that when born, the drone can orient differently, even the target is behind the drone.
    Target is random.
    For the scene, we use the box environment. This box has a open top so we can see the drone flying inside the box.
    Pillar obstacles are randomly placed in the box for different scenes.
    ENV can be the test set and DreamENV can be the training set.
    IDEA: the observation can be one with the projection of the target point on the local frame of the drone.
    We can try with Quaternion and Transition Matrix.

    """
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
        # default state_generator: randomize position within sensing range, yaw randomization
        from math import pi
        # sensing limit: use 10m as default
        limit = 10.0
        if not random_kwargs or "state_generator" not in random_kwargs:
            random_kwargs = {
                "state_generator": {
                    "class": "Uniform",
                    "kwargs": [
                        {"position": {"mean": [0., 0., 1.], "half": [limit, limit, 0.]} }
                    ]
                }
            }
        # apply yaw randomization to orientation half-angle
        for kw in random_kwargs["state_generator"]["kwargs"]:
            kw["orientation"] = {"mean": [0., 0., 0.], "half": [0., 0., pi]}
        # domain randomization: add drag noise in random_kwargs
        if "drag_random" not in random_kwargs:
            random_kwargs["drag_random"] = 0.1

        # ensure depth sensor is included
        if not sensor_kwargs:
            sensor_kwargs = [{
                "sensor_type": SensorType.DEPTH,
                "uuid": "depth",
                "resolution": [64, 64],
            }]
        # default to a simple box scene if none provided
        if not scene_kwargs or "path" not in scene_kwargs:
            scene_kwargs = {"path": "VisFly/datasets/spy_datasets/configs/garage_simple"}
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
            target=target,
            max_episode_steps=max_episode_steps
        )
        self.max_sense_radius = 10
        # Use raw depth observations directly (no encoder)
        # Ensure depth in observation space
        self.observation_space["depth"] = spaces.Box(
            low=0.0,
            high=self.max_sense_radius,
            shape=(1, 64, 64),
            dtype=np.float32,
        )
        # reset observation space to state, raw depth, and target (needed by StateTargetImageExtractor)
        self.observation_space = spaces.Dict({
            "state": self.observation_space["state"],
            "depth": self.observation_space["depth"],
            "target": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })

        # ---------------------------------------------------------------------
        # Physics-driven reward configuration (unified for BPTT & evaluation)
        # ---------------------------------------------------------------------
        self.lambda_v = 1.0   # velocity tracking weight
        self.lambda_c = 20.0   # obstacle avoidance weight
        self.lambda_a = 0.1   # acceleration smoothness weight
        self.lambda_j = 0.05  # jerk smoothness weight
        self.max_velocity = 5.0
        self.drone_radius = 0.3

        # Buffers for acceleration / jerk computation
        self.prev_velocity = th.zeros((self.num_envs, 3), device=self.device)
        self.prev_acceleration = th.zeros((self.num_envs, 3), device=self.device)

    def get_observation(self):
        # Ensure consistent device usage
        device = self.device
        position = self.position.to(device)
        target = self.target.to(device)
        
        # relative position to target
        rela_pos = (target - position)
        # distance and relative direction to target  
        distance = rela_pos.norm(dim=1)
        direction = rela_pos / (distance.unsqueeze(1) + 1e-6)
        
        # velocity and angular velocity
        velocity = self.velocity.to(device)
        angular_velocity = self.angular_velocity.to(device)
        
        state = th.cat([
            rela_pos,
            direction, 
            velocity,
            angular_velocity,
            distance.unsqueeze(1),
        ], dim=1).to(device)
        
        # Add depth sensor observation if visual is enabled
        if self.visual and "depth" in self.sensor_obs:
            depth = th.from_numpy(self.sensor_obs["depth"]).to(device)
            # Normalize depth to [0, 1] range
            depth = (depth / self.max_sense_radius).clamp(0, 1)
        else:
            # Provide zero depth if no sensor data available
            depth = th.zeros((self.num_envs, 1, 64, 64), device=device)
        
        # Return TensorDict as expected by the base class
        from VisFly.utils.type import TensorDict
        return TensorDict({
            "state": state,
            "depth": depth,
            "target": target.detach() if not self.requires_grad else target,
        })

    def get_failure(self) -> th.Tensor:
        """Episodes end on collision."""
        return self.is_collision

    def get_success(self) -> th.Tensor:
        """Episodes succeed when agent reaches target within threshold."""
        success_threshold = 1.0
        # Ensure device consistency
        position = self.position.to(self.device)
        target = self.target.to(self.device)
        distance_to_target = (position - target).norm(dim=1)
        return distance_to_target < success_threshold

    def reset(self, *args, **kwargs):
        """Randomize target and yaw, then reset agents."""
        # First perform standard reset to get agent start positions
        result = super().reset(*args, **kwargs)
        
        # Now generate targets that are farther from start positions
        limit = self.max_sense_radius  # 10.0
        num = self.num_envs
        
        # Get current agent positions after reset (ensure they're on the right device)
        start_positions = self.position.clone().detach()[:, :2]  # XY only, detached to avoid grad issues
        
        # Generate targets that are at least 3-8 meters away from start positions
        min_distance = 3.0  # minimum distance from start
        max_distance = 8.0  # maximum distance from start
        
        targets_xy = th.zeros((num, 2), device=self.device)
        
        for i in range(num):
            # Generate random direction (angle)
            angle = th.rand(1, device=self.device) * 2 * th.pi
            # Generate random distance between min and max
            distance = min_distance + (max_distance - min_distance) * th.rand(1, device=self.device)
            
            # Calculate target position
            target_x = start_positions[i, 0] + distance * th.cos(angle)
            target_y = start_positions[i, 1] + distance * th.sin(angle)
            
            # Clamp to stay within scene bounds (±limit)
            target_x = th.clamp(target_x, -limit + 1.0, limit - 1.0)
            target_y = th.clamp(target_y, -limit + 1.0, limit - 1.0)
            
            targets_xy[i, 0] = target_x
            targets_xy[i, 1] = target_y
        
        # Keep Z target at same default height
        z = th.ones((num, 1), device=self.device) * 1.0
        self.target = th.cat([targets_xy, z], dim=1)

        # Reset previous velocity/acceleration trackers for physics reward
        with th.no_grad():
            # Reinitialize tracking buffers on the actual simulation device
            sim_dev = self.position.device  # after reset positions are ready
            self.prev_velocity = th.zeros((self.num_envs, 3), device=sim_dev)
            self.prev_acceleration = th.zeros((self.num_envs, 3), device=sim_dev)

        return result

    def get_reward(self):
        # Unified physics-driven reward applicable for both BPTT training and evaluation
        collision_vector = self.collision_vector
        is_collision = self.is_collision
        success = self.get_success()
        reward = self.get_analytical_reward(self, collision_vector, is_collision, success)
        # Ensure reward tensor matches the base class device expectation
        return reward.to(self.device)

    def get_analytical_reward(
            self,
            dyn,
            collision_vector: th.Tensor,
            is_collision: th.Tensor,
            success: th.Tensor,
    ) -> th.Tensor:
        """Physics-driven analytical reward (fully differentiable)."""
        # Use the device of dynamics tensors to avoid mismatch (CPU vs CUDA)
        device = dyn.position.device

        # ------------------------------------------------------------------
        # Safeguard against autograd version mismatch
        # ------------------------------------------------------------------
        # Clone tensors that will be mutated by the simulator in later timesteps
        # so the reward graph keeps its own immutable copies.
        pos = dyn.position.clone()
        vel = dyn.velocity.clone()
        dir_vec = dyn.direction.clone() if hasattr(dyn, "direction") else None

        # Ensure collision_vector lives on the same device and is an independent copy
        collision_vector = collision_vector.to(device).clone()

        # Move tracking buffers to correct device if needed
        if self.prev_velocity.device != device:
            self.prev_velocity = self.prev_velocity.to(device)
        if self.prev_acceleration.device != device:
            self.prev_acceleration = self.prev_acceleration.to(device)

        # ------------------------------------------------------------------
        # 1. Velocity tracking toward target
        # ------------------------------------------------------------------
        target_dir = (self.target.to(device) - pos)
        target_dis = target_dir.norm(dim=1, keepdim=True)
        target_dir_normalized = target_dir / (target_dis + 1e-6)
        target_vel_mag = (target_dis.squeeze() * 2.0).clamp(0.5, self.max_velocity)
        target_velocity = target_dir_normalized * target_vel_mag.unsqueeze(1)
        velocity_error = (vel - target_velocity).norm(dim=1)
        r_velocity = -self.lambda_v * velocity_error

        # ------------------------------------------------------------------
        # 2. Obstacle avoidance (distance + approaching speed)
        # ------------------------------------------------------------------
        # Ensure indicator tensors share the same device
        is_collision = is_collision.to(device)
        success = success.to(device)

        collision_dis = collision_vector.norm(dim=1).clamp(min=1e-6)
        approaching_v, _, _ = get_along_vertical_vector(collision_vector, vel)
        approaching_v = approaching_v.clamp(min=0)  # only penalise when moving towards obstacle
        obstacle_penalty = th.where(
            collision_dis < (self.drone_radius * 3),
            (self.drone_radius / collision_dis) * approaching_v,
            th.zeros_like(collision_dis)
        )
        r_obstacle = -self.lambda_c * obstacle_penalty

        # ------------------------------------------------------------------
        # 3. Control smoothness (acceleration & jerk)
        # ------------------------------------------------------------------
        curr_acc = (vel - self.prev_velocity) / 0.02  # dt hard-coded as in dynamics
        r_acc = -self.lambda_a * curr_acc.norm(dim=1)

        jerk = curr_acc - self.prev_acceleration
        r_jerk = -self.lambda_j * jerk.norm(dim=1)

        # Update buffers without tracking gradients
        with th.no_grad():
            # Reassign new detached tensors to avoid in-place modification issues
            self.prev_velocity = vel.detach().clone()
            self.prev_acceleration = curr_acc.detach().clone()

        # ------------------------------------------------------------------
        # 4. Terminal terms: success & collision
        # ------------------------------------------------------------------
        r_success = success.float() * 10.0
        r_collision = is_collision.float() * -5.0

        total_reward = r_velocity + r_obstacle + r_acc + r_jerk + r_success + r_collision
        return total_reward.to(device)