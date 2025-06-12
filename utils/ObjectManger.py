import numpy as np
import torch as th
from ..utils.randomization import UniformStateRandomizer, NormalStateRandomizer, StateRandomizer
import os, sys
import json

g = th.as_tensor([[0, 0, -9.8]])


class ObjectManager:
    def __init__(
            self,
            scene_handle,
            dt,
            object_scene_handle=None,
            object_path=None,
            isolated=False,
            device=th.device("cpu")

    ):
        """

        Args:
            num_scene: num of scenes
            dt: time interval
            object_scene_handle: object handles in habitat-sim if vision is available
        """
        self.num_scene = sc
        self.handles = object_scene_handle
        self.dt = dt

        self._position = None
        self._orientation = None
        self._velocity = None
        self._angular_velocity = None

        self._object_isolated = isolated
        self._object_path = object_path
        self._init_model()

        self.device = device

    def _init_model(self):
        current_file_addr = os.path.dirname(os.path.abspath(__file__))
        _object_setting = current_file_addr + "/../configs/objects.json" if self._object_path is None else self._object_path
        js_file = open(_object_setting)
        kwargs = json.load(js_file)["objects"]
        state_generators = []
        for kwarg in kwargs:
            if kwarg["class"] == "uniform":
                state_generators.append(UniformStateRandomizer(**kwarg["kwargs"]))
            elif kwargs["class"] == "normal":
                state_generators.append(NormalStateRandomizer(**kwarg["kwargs"]))

        self.state_generators = state_generators
        if self.num_scene > 1 or len(kwargs) > 1:
            raise NotImplementedError
        self.reset()

    def step(self):
        self._velocity, self._angular_velocity = self.compute_spd()
        self._position += self._velocity * self.dt

    def reset_by_id(self, indices=None):
        if indices is None:
            self.reset()
        else:
            if self._object_isolated:
                pos, ori, vel, ang_vel = self.state_generators[0](num=len(indices))

            else:
                raise NotImplementedError

            self._position[indices] = pos
            self._orientation[indices] = ori
            self._velocity[indices] = vel
            self._angular_velocity[indices] = ang_vel

    def compute_spd(self):
        velocity = self.velocity + g*self.dt
        angular_velocity = th.zeros((self.num_envs, 3))
        return velocity, angular_velocity

    def reset(self):
        if self._object_isolated:
            pos, ori, vel, ang_vel = self.state_generators[0](num=self.num_agent_per_scene)

        else:
            raise NotImplementedError
        self._position = pos
        self._orientation = ori
        self._velocity = vel
        self._angular_velocity = ang_vel

    @property
    def position(self):
        return self._position

    @property
    def orientation(self):
        return self._orientation

    @property
    def velocity(self):
        return self._velocity

    @property
    def angular_velocity(self):
        return self._angular_velocity

    @property
    def state(self):
        return th.cat([self.position, self.orientation, self.velocity, self.angular_velocity], dim=1)
