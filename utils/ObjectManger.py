import habitat_sim
import numpy as np
import torch as th
import magnum as mn

from .common import std_to_habitat
from ..utils.randomization import UniformStateRandomizer, NormalStateRandomizer, StateRandomizer, load_generator, load_dist
from ..utils.datasets.datasets import get_files_with_suffix
import os, sys
import json

g = th.as_tensor([[0, 0, -9.8]])


class Path:
    def __init__(self, cls, velocity, kwargs):
        self.cls = cls
        self.velocity = velocity
        if "comment" in kwargs:
            kwargs.pop("comment")
        if cls == "circle":
            self.radius = kwargs["radius"]
            self.center = kwargs["center"]
            self.angular_velocity = self.velocity / self.radius
        elif cls == "polygon":
            self.points = th.as_tensor(kwargs["points"])
            self.num_points = len(self.points)
            assert self.num_points >= 2, "Polygon path must have at least two points."
            self.current_index = 0
            self.position = self.points[0]

    # description
    def __repr__(self):
        return f"ObjectManager(type={self.cls})"

    def get_target(self, t, pos, dt):
        if self.cls == "circle":
            x = self.radius * np.cos(self.angular_velocity * t) + self.center[0]
            y = self.radius * np.sin(self.angular_velocity * t) + self.center[1]
            z = self.center[2]
            return th.tensor([x, y, z], dtype=th.float32).unsqueeze(0)
        elif self.cls == "polygon":
            next_index = (self.current_index + 1) % self.num_points
            dis_vec = self.points[next_index] - self.position
            dis_norm = dis_vec.norm()
            dis_vector = dis_vec / dis_norm
            velocity = self.velocity if dis_norm > self.velocity * dt else dis_norm / dt
            self.position = self.position + dis_vector * velocity * dt
            # print(dis_norm,  self.velocity * dt)

            if dis_norm <= self.velocity * dt:
                self.current_index = next_index
            return self.position
        elif self.cls == "rrt":
            pass

    def reset(self):
        if self.cls == "circle":
            pass
        elif self.cls == "polygon":
            self.current_index = 0
        elif self.cls == "rrt":
            pass
            # replan the path


class ObjectManager:
    def __init__(
            self,
            obj_mgr,
            dt,
            path=None,
            scene_id=None,
            collision_func=None,
            scene_node=None,
    ):
        """
        Args:
            dt: time interval
        """
        self.obj_mgr = obj_mgr
        self.dt = dt

        self._t = 0

        self._path = path
        self.scene_node = scene_node
        self._init_model(scene_id=scene_id, collision_func=collision_func)

    def _load_data_generator(self, data):
        pass

    def _init_model(self, scene_id=None, collision_func=None):
        js_file = open(self._path)
        objs_setting = json.load(js_file)["objects"]

        self._generators = []
        self._velocity = []
        self._angular_velocity = []

        self._positions = [None for _ in range(len(objs_setting))]
        self._orientations = [None for _ in range(len(objs_setting))]

        self._objects_handles = []
        self._target_mgrs = []

        root_addr = os.path.dirname(__file__) + "/../"

        # self._model_paths = []
        for obj_setting in objs_setting:
            name = obj_setting["name"]
            obj_model_path = root_addr + f'datasets/visfly-beta/configs/objects/{obj_setting["model_path"]}'
            model_path = get_files_with_suffix(obj_model_path, ".json")
            generator = load_generator(
                cls=obj_setting["initial"]["class"],
                kwargs=obj_setting["initial"]["kwargs"],
                scene_id=scene_id,
                is_collision_func=collision_func
            )
            self._generators.append(generator)

            velocity = load_dist(obj_setting["velocity"]).generate(1)
            self._velocity.append(velocity)
            self._angular_velocity.append(load_dist(obj_setting["angular_velocity"]).generate(1))
            self._objects_handles.append(
                self.obj_mgr.add_object_by_template_handle(model_path[th.randint(0, len(model_path), (1,))],attachment_node=self.scene_node)
            )
            self._objects_handles[0].motion_type = habitat_sim.physics.MotionType.DYNAMIC

            self._target_mgrs.append(Path(cls=obj_setting["path"]["class"], velocity=velocity, kwargs=obj_setting["path"]["kwargs"]))

        self.reset()

    def reset(self):
        self._t = 0
        for i in range(len(self._objects_handles)):
            pos, ori, vel, ang_vel = self._generators[i].safe_generate(num=1)
            self._positions[i] = pos[0]
            self._orientations[i] = ori[0]
            self._target_mgrs[i].reset()
        self.step()

    def step(self):
        self._t += self.dt
        for i in range(len(self._objects_handles)):
            position = self._target_mgrs[i].get_target(t=self._t, dt=self.dt, pos=self._positions[i])
            self._positions[i] = position
            # dis = (position-self._positions[i]).norm()
            # self._positions[i], self._orientations[i] = position, orientation
            # hab_pos, hab_ori = std_to_habitat(position, orientation)
            hab_pos, _ = std_to_habitat(position, None)
            self._objects_handles[i].root_scene_node.translation = mn.Vector3(hab_pos[0])
            # self._objects_handles[i].root_scene_node.translate(hab_pos[0])

            # obj.root_scene_node.transformation =
            # obj.scene_node.rotation = mn.Quaternion(mn.Vector3(hab_ori[1:]), hab_ori[0])

    @property
    def position(self):
        return th.vstack(self._positions)

    @property
    def orientation(self):
        return th.tensor(self._orientations)

    @property
    def velocity(self):
        return th.tensor(self._velocity)

    @property
    def angular_velocity(self):
        return self._angular_velocity

    @property
    def state(self):
        return th.cat([self.position, self.orientation, self.velocity, self.angular_velocity], dim=1)
