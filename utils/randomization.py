import torch as th
from .type import Uniform, Normal
from typing import Union, Optional, Dict
from .maths import Quaternion
from abc import abstractmethod

rotation_matrices = th.tensor([
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 0°
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],  # 90°
    [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],  # 180°
    [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]  # 270°
], dtype=th.float32)


class StateRandomizer:
    def __init__(self,
                 position,
                 orientation,  # euler angle
                 velocity,
                 angular_velocity,
                 seed: int = 42,
                 is_collision_func: Optional[callable] = None,
                 scene_id: Optional[int] = None,
                 device: th.device = th.device("cpu")
                 ):

        self.position = position
        self.orientation = orientation
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.is_collision_func = is_collision_func
        self.device = device
        self.scene_id = scene_id
        # self.set_seed(seed)

    @abstractmethod
    def _generate(self, num) -> tuple:
        pass

    def generate(self, num, **kwargs):
        raw_pos, raw_ori, raw_vel, raw_ang_vel = self._generate(num,  **kwargs)
        return raw_pos, raw_ori, raw_vel, raw_ang_vel

    def safe_generate(self, num=1, **kwargs):
        raw_pos, raw_ori, raw_vel, raw_ang_vel = self.generate(num,  **kwargs)
        position = raw_pos
        orientation = raw_ori
        velocity = raw_vel
        angular_velocity = raw_ang_vel
        # position = (2 * th.rand(num, 3) - 1) * self.position.half + self.position.mean
        # orientation = (2 * th.rand(num, 3) - 1) * self.orientation.half + self.orientation.mean
        # velocity = (2 * th.rand(num, 3) - 1) * self.velocity.half + self.velocity.mean
        # angular_velocity = (2 * th.rand(num, 3) - 1) * self.angular_velocity.half + self.angular_velocity.mean

        if self.is_collision_func is not None:
            is_collision = self.is_collision_func(std_positions=position, scene_id=self.scene_id)
            while True:
                if not is_collision.any():
                    break
                raw_pos, raw_ori, raw_vel, raw_ang_vel = self.generate(is_collision.sum(),  **kwargs)
                position[is_collision, :] = raw_pos
                orientation[is_collision, :] = raw_ori
                velocity[is_collision, :] = raw_vel
                angular_velocity[is_collision, :] = raw_ang_vel
                # position[is_collision, :] = (2 * th.rand(is_collision.sum(), 3) - 1) * self.position.half + self.position.mean
                # orientation[is_collision, :] = (2 * th.rand(is_collision.sum(), 3) - 1) * self.orientation.half + self.orientation.mean
                # velocity[is_collision, :] = (2 * th.rand(is_collision.sum(), 3) - 1) * self.velocity.half + self.velocity.mean
                # angular_velocity[is_collision, :] = (2 * th.rand(is_collision.sum(), 3) - 1) * self.angular_velocity.half + self.angular_velocity.mean
                is_collision = self.is_collision_func(std_positions=position, scene_id=self.scene_id)

        orientation = Quaternion.from_euler(*orientation.T).toTensor().T
        return position.to(self.device), orientation.to(self.device), velocity.to(self.device), angular_velocity.to(self.device)

    def set_seed(self, seed=42):
        th.manual_seed(seed)

    def to(self, device):
        self.device = device
        return self


class UniformStateRandomizer(StateRandomizer):
    def __init__(self,
                 position={"mean": [0., 0., 0.], "half": [0., 0., 0.]},
                 orientation={"mean": [0., 0., 0.], "half": [0., 0., 0.]},  # euler angle
                 velocity={"mean": [0., 0., 0.], "half": [0., 0., 0.]},
                 angular_velocity={"mean": [0., 0., 0.], "half": [0., 0., 0.]},
                 seed: int = 42,
                 is_collision_func: Optional[callable] = None,
                 scene_id: Optional[int] = None,
                 device: th.device = th.device("cpu")
                 ):
        super().__init__(
            position=position,
            orientation=orientation,
            velocity=velocity,
            angular_velocity=angular_velocity,
            seed=seed,
            is_collision_func=is_collision_func,
            scene_id=scene_id,
            device=device
        )

        self.position = Uniform(**position)
        self.orientation = Uniform(**orientation)
        self.velocity = Uniform(**velocity)
        self.angular_velocity = Uniform(**angular_velocity)

    def _generate(self, num, **kwargs) -> tuple:
        position = (2 * th.rand(num, *self.position.mean.shape) - 1) * self.position.half.unsqueeze(0) + self.position.mean.unsqueeze(0)
        orientation = (2 * th.rand(num, *self.position.mean.shape) - 1) * self.orientation.half.unsqueeze(0) + self.orientation.mean.unsqueeze(0)
        velocity = (2 * th.rand(num, *self.position.mean.shape) - 1) * self.velocity.half.unsqueeze(0) + self.velocity.mean.unsqueeze(0)
        angular_velocity = (2 * th.rand(num, *self.position.mean.shape) - 1) * self.angular_velocity.half.unsqueeze(0) + self.angular_velocity.mean.unsqueeze(0)
        return position, orientation, velocity, angular_velocity


class NormalStateRandomizer(StateRandomizer):
    def __init__(
            self,
            position={"mean": [0., 0., 0.], "std": [0., 0., 0.]},
            orientation={"mean": [0., 0., 0.], "std": [0., 0., 0.]},  # euler angle
            velocity={"mean": [0., 0., 0.], "std": [0., 0., 0.]},
            angular_velocity={"mean": [0., 0., 0.], "std": [0., 0., 0.]},
            seed: int = 42,
            is_collision_func: Optional[callable] = None,
            scene_id: Optional[int] = None,
            device: th.device = th.device("cpu")
    ):
        super().__init__(
            position=position,
            orientation=orientation,
            velocity=velocity,
            angular_velocity=angular_velocity,
            seed=seed,
            is_collision_func=is_collision_func,
            scene_id=scene_id,
            device=device
        )

        self.position = Normal(**position)

    def _generate(self, num, **kwargs) -> tuple:
        position = (2 * th.randn(num, *self.position.mean.shape) - 1) * self.position.std.unsqueeze(0) + self.position.mean.unsqueeze(0)
        orientation = (2 * th.randn(num, *self.position.mean.shape) - 1) * self.orientation.std.unsqueeze(0) + self.orientation.mean.unsqueeze(0)
        velocity = (2 * th.randn(num, *self.position.mean.shape) - 1) * self.velocity.std.unsqueeze(0) + self.velocity.mean.unsqueeze(0)
        angular_velocity = (2 * th.randn(num, *self.position.mean.shape) - 1) * self.angular_velocity.std.unsqueeze(0) + self.angular_velocity.mean.unsqueeze(0)
        return position, orientation, velocity, angular_velocity


class TargetUniformRandomizer(UniformStateRandomizer):
    def __init__(self, min_dis=0.5, max_dis=10.0, test=False, *args, **kwargs):
        self.min_dis = min_dis
        self.max_dis = max_dis
        self.test = test
        if self.test:
            self.current_generate_index = 0
        super().__init__(*args, **kwargs)
        
    def _generate(self, num, **kwargs) -> tuple:
        def calculate_yaw_pitch(vector):
            """s
            Calculate the yaw and pitch angles of a vector.

            Args:
                vector (np.ndarray): A 3D vector [x, y, z].

            Returns:
                tuple: (yaw, pitch) in radians.
            """
            x, y, z = vector[:, 0], vector[:, 1], vector[:, 2]

            # Calculate yaw (arctan2 handles the quadrant correctly)
            yaw = th.arctan2(y, x)
            yaw = th.arccos(x / vector[:,:2].norm(dim=1)) * y.sign()
            # Calculate pitch
            norm = th.linalg.norm(vector)  # Magnitude of the vector
            pitch = th.arcsin(z / norm)
            return yaw, pitch
        target_position = kwargs["position"]
        if not self.test:
            position = ((2 * th.rand(num, *self.position.half.shape) - 1) * self.position.half.unsqueeze(0))
        else:
            position = th.tile(kwargs["velocity"].unsqueeze(0), (num, 1))
            position = (rotation_matrices[self.current_generate_index % 4] @ position.T).T
            self.current_generate_index += 1

        position_norm = position.norm(dim=1, keepdim=True)
        # Create scaling factor
        scale_factor = th.ones_like(position_norm)
        # If norm > max_dis, scale down
        scale_factor = th.where(position_norm > self.max_dis, self.max_dis / position_norm, scale_factor)
        # If norm < min_dis, scale up
        scale_factor = th.where(position_norm < self.min_dis, self.min_dis / position_norm, scale_factor)
        # Apply scaling
        position = position * scale_factor
        position = position + target_position.unsqueeze(0)
        direction = target_position.unsqueeze(0)-position
        yaw, pitch = calculate_yaw_pitch(direction)
        orientation = th.stack([th.zeros(num), pitch*0, yaw], dim=1) + (2 * th.rand(num, 3) - 1) * self.orientation.half # yaw, pitch, roll
        if "velocity" in kwargs.keys():
            velocity = th.tile(kwargs["velocity"].unsqueeze(0), (num, 1))
        else:
            velocity = (2 * th.rand(num, 3) - 1) * self.velocity.half + self.velocity.mean
        angular_velocity = (2 * th.rand(num, 3) - 1) * self.angular_velocity.half + self.angular_velocity.mean

        return position, orientation, velocity, angular_velocity

class UnionRandomizer:
    Randomizer_alias = {
        "Uniform": UniformStateRandomizer,
        "Normal": NormalStateRandomizer
    }

    def __init__(
            self,
            randomizers_kwargs: list,
            device,
            is_collision_func=None,
            scene_id=0,
    ):
        self.randomizers = []
        for randomizers in randomizers_kwargs:
            self.randomizers.append(
                self.Randomizer_alias[randomizers["class"]](
                    device=device,
                    is_collision_func=is_collision_func,
                    scene_id=scene_id,
                    **randomizers["kwargs"]
                )
            )

    def __Len__(self):
        return len(self.randomizers)

    def to(self, device):
        for randomizer in self.randomizers:
            randomizer.to(device)

    def _generate(self, num) -> tuple:
        position, orientation, velocity, angular_velocity = [], [], [], []
        for randomizer in self.randomizers:
            pos, ori, vel, ang_vel = randomizer.generate(num)
            position.append(pos)
            orientation.append(ori)
            velocity.append(vel)
            angular_velocity.append(ang_vel)

        position, orientation, velocity, angular_velocity = th.stack(position), th.stack(orientation), th.stack(velocity), th.stack(angular_velocity)
        select_randomizer_index = th.randint(0, len(self.randomizers), (num,))
        row = th.arange(num)
        return position[row, select_randomizer_index], orientation[row, select_randomizer_index], velocity[row, select_randomizer_index], angular_velocity[row, select_randomizer_index]

    def safe_generate(self, num) -> tuple:
        position, orientation, velocity, angular_velocity = [], [], [], []
        for randomizer in self.randomizers:
            pos, ori, vel, ang_vel = randomizer.safe_generate(num)
            position.append(pos)
            orientation.append(ori)
            velocity.append(vel)
            angular_velocity.append(ang_vel)

        position, orientation, velocity, angular_velocity = th.stack(position), th.stack(orientation), th.stack(velocity), th.stack(angular_velocity)
        select_randomizer_index = th.randint(0, len(self.randomizers), (num,))
        row = th.arange(num)
        return position[select_randomizer_index, row, :], orientation[select_randomizer_index, row, :], velocity[select_randomizer_index, row, :], angular_velocity[select_randomizer_index, row, :]


def load_generator(cls, kwargs, is_collision_func=None, scene_id=None, device="cpu"):
    cls_alias = {
        "Uniform": UniformStateRandomizer,
        "Normal": NormalStateRandomizer,
        "Union": UnionRandomizer,
        "TargetUniform": TargetUniformRandomizer
    }

    if isinstance(cls, str):
        cls = cls_alias[cls]

    return cls(is_collision_func=is_collision_func, scene_id=scene_id, device=device, **kwargs)


def load_dist(data):
    cls_alias = {
        "Uniform": Uniform,
        "Normal": Normal,
    }
    if not isinstance(data, dict):
        kwargs = {
            "mean": data,
            "half": 0.
        }
        cls = Uniform
    else:
        cls = cls_alias[data["class"]]
        kwargs = data["kwargs"]
    return cls(**kwargs)