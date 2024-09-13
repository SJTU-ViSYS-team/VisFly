import habitat_sim
import os
import sys
import torch

sys.path.append(os.getcwd())

from itertools import cycle

from .dataloader import SimpleDataLoader
from torch.utils.data import DataLoader
from torch import Tensor
from typing import Union, List, Tuple, Type
from .pathFinder import PRMPlanner
import numpy as np

from scipy.spatial.transform import Rotation as R
from .datasets import ChildrenPathDataset

from typing import List, Union, Optional
import quaternion
import magnum as mn
from .common import *
from abc import ABC

DEBUG = False

# parameters definition
origin = mn.Vector3(0.0, 0.0, 0.0)
eye_pos0 = mn.Vector3(0.5, 0.2, 0.1)
eye_pos1 = mn.Vector3(3.5, 3.0, 4.5)
eye_pos_follow = mn.Vector3(-0.1, -0.2, -0.5)
eye_pos_near = mn.Vector3(0.1, 0.5, 1)*3
eye_pos_back = mn.Vector3(0, 0.8, 2)

eye_pos_follow_back = mn.Vector3(0, 0.5, 1)
eye_pos_follow_near = mn.Vector3(0.1, 0.5, 1)
# eye_pos_follow_near = mn.Vector3(.1, 1, -0.5)

copacity = 1.0
# BGR
# ColorSet = [
#     mn.Color4(45./255, 45./255, 45./255, copacity),
#     mn.Color4(255./255, 96./255, 0./255, copacity),
#     mn.Color4(255./255, 165./255, 89./255, copacity),
#     mn.Color4(255./255, 230./255, 199./255, copacity),
# ]
ColorSet = [
    mn.Color4(69. / 255, 69. / 255, 69. / 255, copacity),  # default color
    mn.Color4(0. / 255, 96. / 255, 255. / 255, copacity),  # highlight color1
    mn.Color4(89. / 255, 165. / 255, 255. / 255, copacity),  # highlight color2
    mn.Color4(199. / 255, 230. / 255, 255. / 255, copacity),  # highlight color3
]

ColorSet2 = [
    mn.Color4(234. / 255, 223. / 255, 180. / 255, copacity),  # default color
    mn.Color4(155. / 255, 176. / 255, 193. / 255, copacity),  # highlight color1
    mn.Color4(81. / 255, 130. / 255, 246. / 255, copacity),  # highlight color2
    mn.Color4(246. / 255, 153. / 255, 92. / 255, copacity),  # highlight color3
]


# create a 20 length similar Colorset using orange as primary color
ColorSet3 = []
for i in np.linspace(0, 1, 100):
    color = mn.Color4(1.0 - 0.5 * i, 0.5 * i, 0.5 * i, 1.0)  # RGBA color
    ColorSet3.append(color)


ColorSet4 = [
    mn.Color4(33. / 255, 156. / 255, 144. / 255, 1.),  # default color
    mn.Color4(233. / 255, 184. / 255, 36. / 255, copacity),  # highlight color1
    mn.Color4(238. / 255, 147. / 255, 34. / 255, copacity),  # highlight color2
    mn.Color4(216. / 255, 63. / 255, 49. / 255, copacity),  # highlight color3
] # rgb

ColorSet4 = [
    mn.Color4(144. / 255,156. / 255,  33. / 255, 0.5),  # default color
    mn.Color4( 36. / 255, 184. / 255, 233. / 255,copacity),  # highlight color1
    mn.Color4(34. / 255, 147. / 255, 238. / 255, copacity),  # highlight color2
    mn.Color4(49. / 255, 63. / 255, 216. / 255, copacity),  # highlight color3
] # bgr


red = mn.Color4(1.0, 0.0, 0.0, copacity)
green = mn.Color4(0.0, 1.0, 0.0, copacity)
blue = mn.Color4(0.0, 0.0, 1.0, copacity)
white = mn.Color4(1.0, 1.0, 1.0, copacity)
orange = mn.Color4(1.0, 0.5, 0.0, copacity)


def color_consequence(
        color1=ColorSet4[2],
        color2=ColorSet4[0],
        factor=1,
):
    factor = np.array(factor).clip(min=0.,max=1.)

    return color1*(1-factor)+factor*color2


def calc_camera_transform(
        eye_translation=mn.Vector3(1, 1, 1), lookat=mn.Vector3(0, 0, 0)
):
    # choose y-up to match Habitat's y-up convention
    camera_up = mn.Vector3(0.0, 1.0, 0.0)
    # camera_up = mn.Vector3(0.0, 0.0, -1.0)

    return mn.Matrix4.look_at(mn.Vector3(eye_translation), mn.Vector3(lookat), camera_up)


class SceneManager(ABC):
    def __init__(
            self,
            path: str = "datasets/spy_datasets/configs/garage_empty",
            scene_type: str = "json",
            num_scene: int = 1,
            num_agent_per_scene: Union[int, List[int]] = 1,
            seed: int = 1,
            uav_radius=0.1,
            sensitive_radius=5,
            semantic=False,
            is_find_path: bool = False,
            multi_drone: bool = False,
            sensor_settings=None,
            render_settings=None,
            reset_settings=None,
            noise_settings=None,
    ):

        if reset_settings is None:
            reset_settings = {
                "pos_rand": False,
                "ori_rand": False,
                "pos_rand_r": None,
            }

        if sensor_settings is None:
            sensor_settings = [{
                "sensor_type": habitat_sim.SensorType.DEPTH,
                "uuid": "depth",
                "resolution": [64, 64],
            }]

        self._get_datasets_info(path)
        self.root_path = path
        self.num_scene = num_scene
        self.num_agent_per_scene = num_agent_per_scene
        self.num_agent = self.num_agent_per_scene * self.num_scene
        self.seed = seed
        self.sensor_settings = sensor_settings
        self.render_settings = render_settings
        self.reset_settings = reset_settings
        self.noise_settings = noise_settings

        self.drone_radius = uav_radius
        self.sensitive_radius = sensitive_radius

        # self._dataLoader = DataLoader(
        #     ChildrenPathDataset(self.root_path, type=scene_type, semantic=semantic), batch_size=num_scene, shuffle=True
        # , num_workers=0)
        self._dataLoader = SimpleDataLoader(
            ChildrenPathDataset(self.root_path, type=scene_type, semantic=semantic), batch_size=num_scene, shuffle=True
        )
        self._scene_loader = cycle(self._dataLoader)
        self.scenes: List[habitat_sim.scene] = [None for _ in range(num_scene)]
        self.agents: List[List[habitat_sim.agent]] = [[] for _ in range(num_scene)]

        self._scene_bounds = [None for _ in range(num_scene)]

        self.is_find_path = is_find_path
        if is_find_path:
            self.path_finders = [PRMPlanner(
                bounds=((0, 10), (0, 10), (0, 10)),
                num_samples=500,
                obstacle_func=lambda point: False,
            ) for _ in range(num_scene)]

        self.is_multi_drone = multi_drone

        if multi_drone:
            self._obj_mgrs: habitat_sim.physics.RigidObjectManager = [None for _ in range(num_scene)]
            self._objects = [[None for _ in range(num_agent_per_scene)] for _ in range(num_scene)]

        if self.render_settings is not None:
            self.render_settings["object_path"] = self.render_settings.get("object_path", self._object_path)
            self.render_settings["line_width"] = self.render_settings.get("line_width", 1.0)
            self.render_settings["axes"] = self.render_settings.get("axes", False)
            self.render_settings["trajectory"] = self.render_settings.get("trajectory", False)
            self.render_settings["sensor_type"] = self.render_settings.get("sensor_type", habitat_sim.SensorType.COLOR)
            self.render_settings["mode"] = self.render_settings.get("mode", "fix")
            self.render_settings["view"] = self.render_settings.get("view", "near")
            self.render_settings["resolution"] = self.render_settings.get("resolution", [256, 256])
            self.render_settings["position"] = self.render_settings.get("position", None)

            self._render_camera = [None for _ in range(num_scene)]
            self._obj_mgrs: habitat_sim.physics.RigidObjectManager = [None for _ in range(num_scene)]
            self._line_renders: habitat_sim.gfx.DebugLineRender = [None for _ in range(num_scene)]
            self._objects = [[None for _ in range(num_agent_per_scene)] for _ in range(num_scene)]

        self.trajectory = [[[] for _ in range(num_agent_per_scene)] for _ in range(num_scene)]
        self._collision_point = [[None for _ in range(num_agent_per_scene)] for _ in range(num_scene)]
        self._is_out_bounds = [[False for _ in range(num_agent_per_scene)] for _ in range(num_scene)]

        # self.load_scenes()

    def _get_datasets_info(self, path):
        parts = path.split("/")
        index = parts.index("datasets")+1
        root_addr = os.path.dirname(__file__)+"/../"
        self.datasets = parts[index]
        if "hm3d" in parts[index].lower():
            self.datasets_name = "hm3d"
            self._datasets_path = root_addr+"datasets/spy_datasets/spy_datasets.scene_dataset_config.json"
            self._object_path = root_addr+"datasets/spy_datasets/configs/agents/DJI_Mavic_Mini_2.object_config.json"
        elif "spy" in parts[index].lower():
            self.datasets_name = "spy_datasets"
            self._datasets_path = root_addr+"datasets/spy_datasets/spy_datasets.scene_dataset_config.json"
            self._object_path = root_addr+"datasets/spy_datasets/configs/agents/DJI_Mavic_Mini_2.object_config.json"
        elif "hssd" in parts[index].lower():
            self.datasets_name = "hssd-hab"
            self._datasets_path = root_addr+"datasets/hssd-hab/hssd-hab.scene_dataset_config.json"
            self._object_path = root_addr+"datasets/hssd-hab/objects/0/DJI_Mavic_Mini_2.object_config.json"
        elif "mp3d" in parts[index].lower():
            self.datasets_name = "mp3d"
            self._datasets_path = root_addr+"datasets/spy_datasets/spy_datasets.scene_dataset_config.json"
            self._object_path = root_addr+"datasets/spy_datasets/configs/agents/DJI_Mavic_Mini_2.object_config.json"
        else:
            raise ValueError("datasets name is not supported")

    def find_paths(self, targets: Optional[List] = None, indices: Optional[int] = None):
        targets = std_to_habitat(targets, None)[0]
        if indices is None:
            _paths = []
            indices = np.arange(self.num_agent)
            for indice in indices:
                scene_id = indice // self.num_agent_per_scene
                agent_id = indice % self.num_agent_per_scene
                _paths.append(
                    habitat_to_std(
                        self.path_finders[scene_id].plan(
                            start=self.agents[scene_id][agent_id].get_state().position,
                            goal=targets[indice],
                        )
                    )[0]
                )
        else:
            scene_id = indices // self.num_agent_per_scene
            agent_id = indices % self.num_agent_per_scene
            _paths = habitat_to_std(
                self.path_finders[scene_id].plan(
                    start=self.agents[scene_id][agent_id].get_state().position,
                    goal=targets[0],
                )
            )[0]
        return _paths

    def get_pose(self, indices: Union[List, int] = -1):
        """_summary_
            get position and rotation of agents by indice
        Args:
            indices (_type_): _description_

        Returns:
            _type_: _description_
        """
        if indices == -1:
            position = np.empty((self.num_agent_per_scene * self.num_scene, 3))
            rotation = np.empty((self.num_agent_per_scene * self.num_scene, 4))
            for scene_id in range(self.num_scene):
                for agent_id in range(self.num_agent_per_scene):
                    state = self.agents[scene_id][agent_id].get_state()
                    std_pos, std_ori = habitat_to_std(state.position, state.rotation.components)
                    position[scene_id * self.num_agent_per_scene + agent_id] = std_pos
                    rotation[scene_id * self.num_agent_per_scene + agent_id] = std_ori

            return position, rotation
        else:
            if not hasattr(indices, "__iter__"):
                indices = [indices]

            position = np.empty((len(indices), 3))
            rotation = np.empty((len(indices), 4))
            for i, indice in enumerate(indices):
                scene_id = indice // self.num_agent_per_scene
                agent_id = indice % self.num_agent_per_scene
                state = self.agents[scene_id][agent_id].get_state()
                position[i] = state.position
                rotation[i] = state.rotation.components
            position, rotation = habitat_to_std(position, rotation)

            return position, rotation

    def set_pose(self, position, rotation):
        """_summary_
            set position and rotation of agents in each scene
        Args:
            position (_type_): _description_
            rotation (_type_): _description_
        """
        assert (
                len(position) == len(rotation) == self.num_agent
        )

        hab_pos, hab_ori = std_to_habitat(position, rotation)
        drone_id = 0
        for scene_id in range(self.num_scene):
            for agent_id in range(self.num_agent_per_scene):
                # self.agents[scene_id][agent_id].set_state(
                #     habitat_sim.AgentState(
                #         position=hab_pos[drone_id], rotation=quaternion.from_float_array(hab_ori[drone_id])
                #     )
                # )
                self.agents[scene_id][agent_id].scene_node.translation = hab_pos[drone_id]
                self.agents[scene_id][agent_id].scene_node.rotation = mn.Quaternion(mn.Vector3(hab_ori[drone_id][1:]), hab_ori[drone_id][0])

                self.trajectory[scene_id][agent_id].insert(0,
                    np.hstack([hab_pos[drone_id], hab_ori[drone_id]])
                )
                drone_id += 1
                if self.is_multi_drone:
                    self._objects[scene_id][agent_id].root_scene_node.transformation = \
                        self.agents[scene_id][agent_id].scene_node.transformation
                    # self._objects[scene_id][agent_id].scene_node.translation = hab_pos[drone_id]
                    # self._objects[scene_id][agent_id].scene_node.rotation = mn.Quaternion(mn.Vector3(hab_ori[drone_id][1:]), hab_ori[drone_id][0])
        self._update_collision_infos()

    def get_observation(self, indices: Optional[int] = None):
        """_summary_
            get observations outside
        Returns:
            _type_: _description_
        """
        obses = []
        if indices is None:
            for scene in self.scenes:
                agent_observations = scene.get_sensor_observations(
                    list(range(self.num_agent_per_scene))
                )
                obses += list(agent_observations.values())
        else:
            for index in indices:
                agent_id = index % self.num_agent_per_scene
                scene_id = index // self.num_agent_per_scene
                obses.append(self.scenes[scene_id].get_sensor_observations(int(agent_id)))
        return obses

    def _update_collision_infos(self, indices: Optional[List] = None, sensitive_radius: float = None):
        """_summary_test
            update the collision distance of each agent
            
        Args:
            sensitive_radius (float, optional): _description_. Defaults to 5.0.
        """

        indices = np.arange(self.num_agent) if indices is None else indices
        indices = [indices] if not hasattr(indices, "__iter__") else indices

        sensitive_radius = self.sensitive_radius if sensitive_radius is None else sensitive_radius

        for indice in indices:
            scene_id = indice // self.num_agent_per_scene
            agent_id = indice % self.num_agent_per_scene

            # test = self.scenes[scene_id].path_finder
            col_record = self.scenes[scene_id].get_closest_collision_point(
                pt=self.agents[scene_id][agent_id].scene_node.translation,
                max_search_radius=sensitive_radius
            )
            self._collision_point[scene_id][agent_id], self._is_out_bounds[scene_id][agent_id] = col_record.hit_pos, col_record.is_out_bound

        if self.is_multi_drone:
            # if indices is None:
            for scene_id in range(self.num_scene):
                positions = np.array([agent.state.position for agent in self.agents[scene_id]])
                cur_dis = \
                    np.linalg.norm(positions - np.array([self._collision_point[scene_id][agent_id] for agent_id in range(self.num_agent_per_scene)]), axis=1)
                rela_dis = np.diag(np.full(self.num_agent_per_scene, np.inf, dtype=np.float32))
                for i in range(self.num_agent_per_scene):
                    j = i + 1
                    rela_dis[i, j:] = np.linalg.norm(positions[j:] - positions[i])
                rela_dis += rela_dis.T
                min_rela_dis, min_indices = np.min(rela_dis, axis=1), np.argmin(rela_dis, axis=1)
                is_rela_dis_less = min_rela_dis < cur_dis
                for agent_id in np.arange(self.num_agent_per_scene)[is_rela_dis_less]:
                    self._collision_point[scene_id][agent_id] = positions[min_indices[agent_id]]

    def get_point_is_collision(self,
                               std_positions: Optional[Tensor] = None,
                               scene_id: Optional[int] = None,
                               uav_radius: Optional[float] = None,
                               hab_positions: Optional[np.ndarray] = None
                               ):
        """_summary_
            get collision information

        Args:
            uav_radius (float, optional): _description_. Defaults to 0.1.
            std_positions (Tensor): _description_
            scene_id (int, optional): _description_
            hab_positions (th.Tensor, optional): _description_

        Returns:
            _type_: _description_
        """
        uav_radius = self.drone_radius if uav_radius is None else uav_radius

        assert scene_id is not None
        if hab_positions is None:
            hab_positions, _ = std_to_habitat(std_positions, None)
        min_distance = np.empty(len(hab_positions), dtype=np.float32)
        is_in_bounds = np.empty(len(hab_positions), dtype=bool)
        for indice, hab_position in enumerate(hab_positions):
            col_record = \
                self.scenes[scene_id].get_closest_collision_point(
                    pt=hab_position.reshape(3, 1),
                    max_search_radius=uav_radius
                )

            min_distance[indice] = np.linalg.norm((col_record.hit_pos - hab_position))
            is_in_bounds[indice] = not col_record.is_out_bound
        return (min_distance < uav_radius) & is_in_bounds

    def get_collision_point(self, indices=None):
        if indices is None:
            return habitat_to_std(np.array(self._collision_point).reshape((-1, 3)), None)[0]
        else:
            return habitat_to_std(np.array(self._collision_point).reshape((-1, 3))[indices], None)[0]

    def render(
            self,
            is_draw_axes: bool = False,
            points: Optional[Tensor] = None,
            lines: Optional[Tensor] = None,
            curves: Optional[Tensor] = None,
            c_curves: Optional[Tensor] = None,
    ):
        """
        ender and observe the movement of agents

        Args:

        Raises:
            ValueError: _description_
        """

        # draw lines in local coordinate of agent_s or objects
        def draw_axes(sim, translation, axis_len=1.0):
            ENU_coordnates_axes = True
            lr = sim.get_debug_line_render()
            if ENU_coordnates_axes:
                lr.draw_transformed_line(translation, mn.Vector3(0, 0, -axis_len), ColorSet[1])
                lr.draw_transformed_line(translation, mn.Vector3(-axis_len / 2, 0, 0), ColorSet[2])
                lr.draw_transformed_line(translation, mn.Vector3(0, axis_len / 2, 0), ColorSet[3])
            else:
                # draw axes with x+ = red, y+ = green, z+ = blue
                # debug: it should be rgb but bgr presented, i don't know why
                lr.draw_transformed_line(translation, mn.Vector3(axis_len, 0, 0), ColorSet[1])
                lr.draw_transformed_line(translation, mn.Vector3(0, axis_len / 2, 0), ColorSet[2])
                lr.draw_transformed_line(translation, mn.Vector3(0, 0, axis_len / 2), ColorSet[3])

        if self.render_settings is None:
            raise EnvironmentError("render settings is not set")

        # output images
        render_imgs = []

        # debug
        if DEBUG or is_draw_axes or self.render_settings["axes"]:
            for scene_id in range(self.num_scene):
                draw_axes(self.scenes[scene_id], origin, axis_len=1)

        if points is not None:
            points = std_to_habitat(points, None)[0]
            for indice, point in enumerate(points):
                self._line_renders[0].draw_circle(
                    mn.Vector3(point[:3]), radius=0.25, color=ColorSet3[indice]
                )

        if lines is not None:
            for line_id in range(len(lines)):
                line = std_to_habitat(lines[line_id], None)[0]
                self._line_renders[0].draw_transformed_line(
                    mn.Vector3(line[0]), mn.Vector3(line[1]), ColorSet3[line_id]
                )

        if curves is not None:
            for curve in curves:
                curve = std_to_habitat(curve, None)[0]
                curve = [mn.Vector3(point) for point in curve]
                self._line_renders[0].draw_path_with_endpoint_circles(
                    curve, 0.1, white)

        # set the pose of objects or agents in the scene
        for scene_id in range(self.num_scene):
            for agent_id in range(self.num_agent_per_scene):
                self._objects[scene_id][agent_id].root_scene_node.transformation = \
                    self.agents[scene_id][agent_id].scene_node.transformation

        # draw the axes of agents
        if self.render_settings["axes"]:
            for scene_id in range(self.num_scene):
                for agent_id in range(self.num_agent_per_scene):
                    # self._line_renders[scene_id].push_transform(self.agents[scene_id][agent_id].scene_node.transformation)
                    self._line_renders[scene_id].push_transform(
                        self._objects[scene_id][agent_id].transformation
                    )
                    draw_axes(self.scenes[scene_id], origin, axis_len=1)
                    self._line_renders[scene_id].pop_transform()

        # draw the trajectory of agents
        if self.render_settings["trajectory"]:
            for scene_id in range(self.num_scene):
                for agent_id in range(self.num_agent_per_scene):
                    # traj = [mn.Vector3(point[:3]) for point in self.trajectory[scene_id][agent_id]]
                    # if len(traj) > 1:
                    #     self._line_renders[0].draw_path_with_endpoint_circles(
                    #         traj, 0.1, white)
                    for line_id in np.arange(len(self.trajectory[scene_id][agent_id])-1):
                        self._line_renders[scene_id].draw_transformed_line(
                            self.trajectory[scene_id][agent_id][line_id][:3],
                            self.trajectory[scene_id][agent_id][line_id+1][:3],
                            color_consequence(factor=line_id/10),
                        )
                    # trajectory_data = np.array(self.trajectory[scene_id][agent_id])
                    # self._line_renders[scene_id].draw_transformed_line(trajectory_data[:3], self.trajectory[scene_id][agent_id][i+1][:3], white)

        # set the render camera pose
        if self.render_settings["mode"] == "follow":
            if self.render_settings["view"] == "back" or self.render_settings["view"] == "near":
                if self.render_settings["view"] == "back":
                    rela_pos = eye_pos_follow_back
                elif self.render_settings["view"] == "near":
                    rela_pos = eye_pos_follow_near
                for scene_id in range(self.num_scene):
                    obj = self.agents[scene_id][0].get_state().position if self.render_settings["position"] is None else std_to_habitat(self.render_settings["position"], None)[0]
                    camera_pose = calc_camera_transform(
                        eye_translation=(self.agents[scene_id][0].scene_node.transformation * mn.Vector4(rela_pos,1)).xyz,
                        lookat=obj
                    )
                    self._render_camera[scene_id].set_state(
                        habitat_sim.AgentState(
                            rotation=R.from_matrix(
                                np.array(camera_pose)[:3, :3]
                            ).as_quat(),
                            position=camera_pose.translation,
                        )
                    )
            else:
                raise NotImplementedError
                for scene_id in range(self.num_scene):
                    pos, quat = (
                        self.agents[scene_id][0].get_state().position,
                        self.agents[scene_id][0].get_state().rotation,
                    )

                    # adaptive pose in local coordinate
                    ### need further update !!!
                    # fixed pose in local coordinate
                    camera_pose = calc_camera_transform(
                        eye_translation=eye_pos_follow + pos, lookat=pos
                    )
                    self._render_camera[scene_id].set_state(
                        habitat_sim.AgentState(
                            rotation=R.from_matrix(np.array(camera_pose)[:3, :3]).as_quat(),
                            position=camera_pose.translation,
                        )
                    )

        elif self.render_settings["mode"] == "fix":
            if self.render_settings["view"] == "top":
                # fix the camera at the center top of the scene to observe the whole scene
                for scene_id in range(self.num_scene):
                    if self.render_settings["position"] is None:
                        scene_aabb = self._scene_bounds[scene_id]
                        scene_center = (scene_aabb.min + scene_aabb.max) / 2
                        scene_height = (
                                               scene_aabb.max[1] - scene_aabb.min[1]
                                       ) + scene_aabb.max[1] * 2
                    else:
                        scene_center = std_to_habitat(self.render_settings["position"], None)[0][0]
                        scene_height = scene_center[1]

                    self._render_camera[scene_id].set_state(
                        habitat_sim.AgentState(
                            rotation=R.from_euler(
                                "zyx", [90, 0, -90], degrees=True
                            ).as_quat(),
                            # position=mn.Vector3(scene_center[0], scene_height, scene_center[2])
                            position=mn.Vector3(scene_center[0], scene_height, scene_center[2]),
                        )
                    )

            elif self.render_settings["view"] == "near":
                # fix the camera at third person view to observe the agent
                obj = origin if self.render_settings["position"] is None else std_to_habitat(self.render_settings["position"], None)[0].squeeze()
                for scene_id in range(self.num_scene):
                    camera_pose = calc_camera_transform(
                        eye_translation=eye_pos_near + obj, lookat=obj
                    )
                    self._render_camera[scene_id].set_state(
                        habitat_sim.AgentState(
                            rotation=R.from_matrix(
                                np.array(camera_pose)[:3, :3]
                            ).as_quat(),
                            position=camera_pose.translation,
                        )
                    )

            elif self.render_settings["view"] == "side":
                hab_position = std_to_habitat(self.render_settings["position"], None)[0]
                for scene_id in range(self.num_scene):
                    self._render_camera[scene_id].set_state(
                        habitat_sim.AgentState(
                            rotation=R.from_euler(
                                "zyx", [-0, -90, 0], degrees=True
                            ).as_quat(),
                            # position=mn.Vector3(scene_center[0], scene_height, scene_center[2])
                            position=mn.Vector3(np.squeeze(hab_position)),
                        )
                    )

            elif self.render_settings["view"] == "back":
                # fix the camera at third person view to observe the agent
                obj = origin if self.render_settings["position"] is None else std_to_habitat(self.render_settings["position"], None)[0]
                for scene_id in range(self.num_scene):
                    camera_pose = calc_camera_transform(
                        eye_translation=eye_pos_back + obj, lookat=obj
                    )
                    self._render_camera[scene_id].set_state(
                        habitat_sim.AgentState(
                            rotation=R.from_matrix(
                                np.array(camera_pose)[:3, :3]
                            ).as_quat(),
                            position=camera_pose.translation,
                        )
                    )
            elif self.render_settings["view"] == "custom":
                position = std_to_habitat(self.render_settings["position"], None)[0]
                for scene_id in range(self.num_scene):
                    camera_pose = calc_camera_transform(
                        eye_translation=position[0], lookat=position[1]
                    )
                    self._render_camera[scene_id].set_state(
                        habitat_sim.AgentState(
                            rotation=R.from_matrix(
                                np.array(camera_pose)[:3, :3]
                            ).as_quat(),
                            position=camera_pose.translation,
                        )
                    )
            else:
                raise ValueError("Invalid render position.")

        else:
            raise ValueError("Invalid render mode.")

        # get images from render cameras
        for scene_id in range(self.num_scene):
            render_imgs.append(
                self.scenes[scene_id].get_sensor_observations(self.num_agent_per_scene)[
                    "render"
                ]
            )

        return render_imgs

    # initialization of the render agent
    def _load_render_camera(self) -> habitat_sim.agent.AgentConfiguration:
        # render settings
        sensor_cfgs_list = []
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = "render"
        sensor_spec.resolution = self.render_settings["resolution"]
        sensor_spec.position = mn.Vector3([0, 0, 0])
        sensor_spec.sensor_type = self.render_settings["sensor_type"]
        sensor_cfgs_list.append(sensor_spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration(
            radius=0.01,
            height=0.01,
            sensor_specifications=sensor_cfgs_list
        )
        return agent_cfg

    def load_scenes(self):
        """
            load scenes and auto switch to next minibatch of scenes
        """
        scene_paths = next(self._scene_loader)
        cycle_i = 0
        while len(scene_paths) < self.num_scene:
            scene_paths.append(scene_paths[cycle_i])
            cycle_i += 1

        for scene_id, scene_path in enumerate(scene_paths):
            # if scene is not empty, close it and release resource.
            if self.scenes[scene_id] is not None:
                self.scenes[scene_id].close()
            self.scenes[scene_id] = self._load_scene(scene_path)
            self._scene_bounds[scene_id] = self.scenes[scene_id].get_active_scene_graph().get_root_node().cumulative_bb
            if self.is_find_path:
                self.path_finders[scene_id].bounds = self._scene_bounds[scene_id]
                self.path_finders[scene_id].scene_id = scene_id
                self.path_finders[scene_id].obstacle_func = self._
                self.path_finders[scene_id].pre_plan()
            # get agent handles in each scene
            self.agents[scene_id] = [
                self.scenes[scene_id].get_agent(agent_id)
                for agent_id in range(self.num_agent_per_scene)
            ]

            # get render agent handles in each scene
            if self.render_settings is not None:
                self._render_camera[scene_id] = self.scenes[scene_id].get_agent(self.num_agent_per_scene)
                # create line renders and object managers
                self._obj_mgrs[scene_id] = self.scenes[scene_id].get_rigid_object_manager()
                self._line_renders[scene_id] = self.scenes[scene_id].get_debug_line_render()
                self._line_renders[scene_id].set_line_width(self.render_settings["line_width"])
                # create objects in each scene
                for agent_id in range(self.num_agent_per_scene):
                    self._objects[scene_id][agent_id] = self._obj_mgrs[scene_id].add_object_by_template_handle(
                        self.render_settings["object_path"]
                    )

            if self.is_multi_drone:
                if self._objects[scene_id][0] is None:
                    self._obj_mgrs[scene_id] = self.scenes[scene_id].get_rigid_object_manager()
                    for agent_id in range(self.num_agent_per_scene):
                        self._objects[scene_id][agent_id] = self._obj_mgrs[scene_id].add_object_by_template_handle(
                            self._object_path
                        )

        test = 1

    def _load_scene(self, scene_path) -> habitat_sim.Simulator:
        """_summary_
        load single scene with agents
        Args:
            scene_path (_type_): _description_

        Returns:
            habitat_sim.Simulator: _description_
        """
        print(scene_path)
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        sim_cfg.scene_dataset_config_file = \
            self._datasets_path
        # "datasets/replica_cad_dataset/replicaCAD.scene_dataset_config.json"
        sim_cfg.enable_physics = False
        # sim_cfg.scene_id = "None" # debug
        sim_cfg.use_semantic_textures = False  # debug
        # sim_cfg.enable_physics = True # debug
        sim_cfg.load_semantic_mesh = False  # debug
        sim_cfg.force_separate_semantic_scene_graph = False  # debug
        sim_cfg.create_renderer = False
        sim_cfg.enable_gfx_replay_save = True
        sim_cfg.leave_context_with_background_renderer = True
        sim_cfg.random_seed = self.seed

        cfg = habitat_sim.Configuration(
            sim_cfg=sim_cfg,
            agents=self._load_agents(self.num_agent_per_scene),
        )
        env = habitat_sim.Simulator(cfg)
        env.seed(self.seed)

        NavmeshSetting = habitat_sim.NavMeshSettings()
        NavmeshSetting.include_static_objects = True
        env.recompute_navmesh(env.pathfinder, NavmeshSetting)
        return env

    def reset(self, std_positions: Tensor, std_orientations: Tensor) -> Tuple[List, List]:
        """
        Summary: external interference to reset all the agents to the initial state,
        and iterate to the next scenes 
        """
        self.load_scenes()
        self.reset_agents(std_positions, std_orientations)
        raise NotImplementedError

        return reset_pos, reset_ori

    def reset_agents(self, std_positions: Tensor, std_orientations: Tensor, indices: Optional[Tensor] = None, ):
        """
        Summary: external interference to reset all the agents to the initial state
        """
        # indices = [indices] if not hasattr(indices, "__iter__") else indices
        hab_positions, hab_orientations = std_to_habitat(std_positions, std_orientations)
        for indice, hab_position, hab_orientation in zip(np.arange(self.num_agent) if indices is None else indices, hab_positions, hab_orientations):
            scene_id = indice // self.num_agent_per_scene
            agent_id = indice % self.num_agent_per_scene
            self._reset_agent(scene_id, agent_id, hab_position, hab_orientation)
        self._update_collision_infos(indices=indices)

    def _reset_agent(self, scene_id: int, agent_id: int, position: np.ndarray, orientation: np.ndarray):
        """_summary_
            reset the agent.
        Args:
            scene_id (int): _description_
            agent_id (int): _description_
        """
        self.trajectory[scene_id][agent_id] = []

        self.agents[scene_id][agent_id].set_state(
            habitat_sim.AgentState(
                position=position,
                rotation=quaternion.from_float_array(orientation),
            )
        )

    def _load_agents(
            self, num_agent: int
    ) -> List[habitat_sim.agent.AgentConfiguration]:
        """_summary_
            load agent configurations in each environment
        Args:
            num_agent (int): _description_

        Returns:
            List[habitat_sim.agent.AgentConfiguration]: _description_
        """
        agent_cfgs_list = []
        for i in range(num_agent):
            agent_cfgs_list.append(self._load_agent())

        # add render agent
        if self.render_settings is not None:
            agent_cfgs_list.append(self._load_render_camera())

        return agent_cfgs_list

    def _load_agent(self) -> habitat_sim.agent.AgentConfiguration:
        """_summary_
            load single agent configuration
        Returns:
            habitat_sim.agent.AgentConfiguration: _description_
        """
        sensor_cfgs_list = self._load_sensor()
        agent_cfg = habitat_sim.agent.AgentConfiguration(
            radius=0.1,
            height=0.1,
            sensor_specifications=sensor_cfgs_list
        )
        return agent_cfg

    def _load_sensor(self) -> List[habitat_sim.sensor.SensorSpec]:
        """_summary_
            load sensors configuration of each agent
        Returns:
            List[habitat_sim.CameraSensorSpec]: _description_
        """
        sensor_cfgs_list = []
        for i, sensor_cfg in enumerate(self.sensor_settings):
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = sensor_cfg.get("uuid", "color")
            sensor_spec.resolution = sensor_cfg.get("resolution", [128, 128])
            sensor_spec.orientation = mn.Vector3(sensor_cfg.get("orientation", [0., 0, 0]))
            sensor_spec.position = mn.Vector3(sensor_cfg.get("position", [0, 0, -0.2]))
            if type(sensor_cfg.get("sensor_type", habitat_sim.SensorType.COLOR)) != habitat_sim._ext.habitat_sim_bindings.SensorType:
                continue
            sensor_spec.sensor_type = sensor_cfg.get("sensor_type", habitat_sim.SensorType.COLOR)
            if self.noise_settings is not None:
                if sensor_spec.uuid in self.noise_settings.keys():
                    sensor_spec.noise_model = self.noise_settings[sensor_spec.uuid].get("model", "None")
                    sensor_spec.noise_model_kwargs = self.noise_settings[sensor_spec.uuid].get("kwargs", {})
            sensor_cfgs_list.append(sensor_spec)

        return sensor_cfgs_list

    def close(self):
        for scene in self.scenes:
            scene.close()
            break

    @property
    def is_out_bounds(self):
        return th.as_tensor(self._is_out_bounds).reshape(-1)

    @property
    def collision_point(self):
        raise NotImplementedError
        return self._collision_point


def debug_get_test_path():
    delta_pos, delta_ori, delta_ori_euler = (
        np.empty([0, 3]),
        np.empty([0, 4]),
        np.empty([0, 3]),
    )
    point0 = np.array([0, 0, 0])
    point1 = np.array([1, 0, -2])
    point2 = np.array([1, 2, 0])
    point3 = np.array([0, 2, 2])
    point4 = np.array([0, 0, 0])
    points = np.vstack([point0, point1, point2, point3, point4])
    step = 1000
    for i in range(len(points)):
        if i != len(points) - 1:
            delta_pos = np.vstack(
                [delta_pos, np.linspace(points[i], points[i + 1], step)]
            )

    delta_ori_euler = np.vstack(
        # [delta_ori_euler, np.linspace(np.array([0, 0, 0]), np.array([0, 0, np.pi * 2]), step)]
        [delta_ori_euler, np.linspace(np.array([0, 0, 0]), np.array([0, 0, np.pi * 0]), step)]

    )
    delta_ori_euler = np.vstack(
        [
            delta_ori_euler,
            # np.linspace(np.array([np.pi / 2, 0, 0]), np.array([np.pi / 2, np.pi * 2, 0]), step),
            np.linspace(np.array([np.pi / 2, 0, 0]), np.array([np.pi / 2, np.pi * 2 * 0, 0]), step),
        ]
    )
    delta_ori_euler = np.vstack(
        [
            delta_ori_euler,
            # np.linspace(np.array([np.pi, 0, 0]), np.array([3 * np.pi, 0, np.pi * 2]), step),
            np.linspace(np.array([np.pi, 0, 0]), np.array([np.pi, 0, 0]), step),

        ]
    )
    delta_ori_euler = np.vstack(
        [
            delta_ori_euler,
            np.linspace(np.array([3 * np.pi / 2, 0, 0]), np.array([3 * np.pi / 2, 0, 0]), step),
        ]
    )
    delta_ori = R.from_euler("zyx", delta_ori_euler).as_quat()
    delta_ori = delta_ori[:, [3, 0, 1, 2]]

    # step *= 4
    # x = np.linspace(0, 2 * np.pi, step)
    # delta_pos = np.vstack([np.sin(x), np.cos(x), -np.sin(x)]).T
    # delta_ori_euler = np.vstack([x + np.pi / 2 * 3, np.zeros_like(x), np.zeros_like(x)]).T
    # delta_ori = R.from_euler("zyx", delta_ori_euler).as_quat()
    # delta_ori = delta_ori[:, [3, 0, 1, 2]]

    return delta_pos, delta_ori


def debug():
    from utils.randomization import UniformStateRandomizer
    from utils.type import Uniform
    import torch as th

    from utils.common import rgba2rgb, obs_list2array

    num_scene = 1
    num_agent_per_scene = 1
    a = SceneManager(
        num_scene=num_scene,
        num_agent_per_scene=num_agent_per_scene,
        path="datasets/spy_datasets/configs/scenes",
        # path="datasets/spy_datasets/stages3",
        # path="datasets/spy_datasets/configs/garage_simple",
        # scene_type="glb",
        scene_type="json",
        # path="/home/lfx-desktop//files/habitat-sim/habitat_scenes/data/versioned_data/hm3d-1.0/hm3d/minival",
        # path="datasets/spy_datasets/stages",
        render_settings={
            "mode": "fix",
            "view": "near",
            "sensor_type": habitat_sim.SensorType.COLOR,
            "axes": True,
            "trajectory": False,
            "object_path": "datasets/spy_datasets/configs/objects/frl_apartment_shoe_01.object_config.json",
            # "object_path": "datasets/replica_cad_dataset/configs/objects/frl_apartment_shoe_01.object_config.json",
            "line_width": 1.0,
        },
        sensor_settings=[
            {
                "uuid": "depth",
                "resolution": [128, 128],
                "position": [0, 0, -0.2],
                "sensor_type": habitat_sim.SensorType.DEPTH,
            },
        ]
    )
    # for i in range(6):
    #     a.load_scenes()
    #     break
    # a.reset()
    # pos, _ = a.get_pose()
    AverageRandomizers = UniformStateRandomizer(
        # position=Gaussian(mean=np.array([3, 0, 0.5]), std=np.array([1.0, 1.0, 0.0])),
        position=Uniform(mean=np.array([0, 1, 1]), half=np.array([0.5, 0.5, 0.0])),
        # position=Gaussian(mean=np.array([3,1,0.5]), std=np.array([0.5,0.5,0])),
        orientation=Uniform(mean=np.array([0, 0, 0]), half=np.array([0, 0, 0])),
        velocity=Uniform(mean=np.array([0, 0, 0]), half=np.array([0, 0, 0])),
        angular_velocity=Uniform(mean=np.array([0, 0, 0]), half=np.array([0, 0, 0])),
        # is_collision_func=a.get_is_collision,
        scene_id=0,
    )

    delta_pos, delta_ori = debug_get_test_path()
    d_x, d_y = np.diff(delta_pos, axis=0)[:, 0], np.diff(delta_pos, axis=0)[:, 1]

    import cv2, time
    from tqdm import tqdm

    frame_id = 0
    max_step = 100000
    scene_update_step = max_step / 10
    pbar = tqdm(range(100000))

    period = 4000

    a.load_scenes()
    reset_pos, reset_ori, _, _ = AverageRandomizers.safe_generate(num=num_agent_per_scene * num_scene)
    a.reset_agents(std_positions=reset_pos, std_orientations=reset_ori)
    pos, _ = a.get_pose()

    indices = np.zeros((num_scene * num_agent_per_scene), dtype=int)

    sta_time = time.time()
    for i in pbar:
        frame_id += 1

        if i % 1000 < 999:
            x_info = "forward" if d_x[i % period] > 0 else "backward"
            y_info = "left" if d_y[i % period] > 0 else "right"

        if frame_id % 100 == 0:
            fps = frame_id / (time.time() - sta_time)
            frame_id = 0
            sta_time = time.time()
            pbar.set_description(f"FPS: {fps:.2f} {x_info} {y_info}")

        new_pos = th.from_numpy(pos + delta_pos[indices % period])
        new_ori = th.from_numpy(delta_ori[indices % period])
        a.set_pose(
            new_pos,
            new_ori,
        )
        obs = a.get_observation()
        # render_obs = rgba2rgb(a.render(fix_pos=th.tensor([[5, 0, 7]])))
        render_obs = rgba2rgb(a.render())

        # print(a.get_collision_distance())
        # print(a.get_obstacle_position())
        is_collispeds = a.get_is_collision()
        for id, is_collisped in enumerate(is_collispeds):
            if is_collisped:
                print("Collision detected")
                print(a.get_collision_point(), a.get_collision_distance())

                reset_pos, reset_ori, _, _ = AverageRandomizers.safe_generate(num=1)
                a.reset_agents(std_positions=reset_pos, std_orientations=reset_ori, indices=id)
                pos[id], _ = a.get_pose(id)
                indices[id] = 0
            else:
                indices[id] += 1

        # create combined image and show
        camera1_obs = obs
        obs_grid = obs_list2array(camera1_obs, num_scene, num_agent_per_scene)
        render_grid = np.vstack(render_obs)
        cv2.imshow("all Grid", obs_grid)
        cv2.imshow("obs Grid", render_grid)
        cv2.waitKey(1)

    a.close()


if __name__ == "__main__":
    debug()
