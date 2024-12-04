import numpy as np
from .dynamics import Dynamics
from ...utils.ObjectManger import ObjectManager
from ...utils.SceneManager import SceneManager
from typing import List, Union, Tuple, Dict, Optional
from ...utils.randomization import UniformStateRandomizer, NormalStateRandomizer, UnionRandomizer
from ...utils.type import Uniform, Normal
from torch import Tensor
from typing import Optional, Type
import torch as th


IS_BBOX_COLLISION = True


class DroneEnvsBase:
    state_generator_alias = {"Uniform": UniformStateRandomizer, "Normal":NormalStateRandomizer, "Union":UnionRandomizer}

    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            random_kwargs: Optional[Dict] = {},
            dynamics_kwargs: Optional[Dict] = {},
            scene_kwargs: Optional[Dict] = {},
            sensor_kwargs: Optional[Dict] = None,
            uav_radius: float = 0.1,
            sensitive_radius: float = 10.,
            multi_drone: bool = False,
            device: Optional[Type[th.device]] = th.device("cpu"),

    ):
        self.device = device
        self.seed = seed

        self._sensor_obs = {}
        self._is_collision = None
        self._collision_dis = None
        self._collision_point = None
        self._collision_vector = None

        self.uav_radius = uav_radius

        self.visual = visual

        self.noise_settings = random_kwargs.get("noise_kwargs", {})
        self._create_noise_model()
        self.dynamics = Dynamics(
            num=num_agent_per_scene * num_scene,
            seed=seed,
            device=device,
            **dynamics_kwargs
        )

        self.is_multi_drone = multi_drone
        if self.is_multi_drone and (num_agent_per_scene == 1):
            raise ValueError("Num of agents should not be 1 in multi drone env.")

        self.sceneManager: SceneManager = SceneManager(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            uav_radius=uav_radius,
            multi_drone=multi_drone,
            sensitive_radius=sensitive_radius,
            sensor_settings=sensor_kwargs,
            noise_settings=self.noise_settings,
            **scene_kwargs
        ) if visual else None

        # if scene_kwargs.get(["object_kwargs"], None) is not None:
        #     self.objectManager = ObjectManager(
        #         num_scene=num_scene,
        #         num_agent_per_scene=num_agent_per_scene,
        #         dt=dynamics_kwargs.get("dt", 0.02),
        #         object_scene_handle=self.sceneManager.object_list if self.sceneManager is not None else None,
        #         **scene_kwargs.get(["object_kwargs"])
        #     )

        self.stateGenerators = self._create_randomizer(
            random_kwargs
        )
        self._scene_iter = random_kwargs.get("scene_iter", False)

        self._create_bbox()
        self._sensor_list = [sensor["uuid"] for sensor in sensor_kwargs] if sensor_kwargs is not None else []
        self._visual_sensor_list = [s for s in self._sensor_list if "IMU" not in s]
        # self.reset()

        self._eval = False

    def _create_noise_model(self):
        self.noise_settings["IMU"] = self.noise_settings.get("IMU", {
            "model": "UniformNoiseModel",
            "kwargs": {
                "mean": 0,
                "half": 0,
            }
        })
        if self.noise_settings["IMU"]["model"] == "UniformNoiseModel":
            self.noise_settings["IMU"] = Uniform(**self.noise_settings["IMU"].get("kwargs", {}))
        elif self.noise_settings["IMU"]["model"] == "GaussianNoiseModel":
            self.noise_settings["IMU"] = Normal(**self.noise_settings["IMU"].get("kwargs", {}))
        else:
            raise ValueError("IMU Noise model does not exist.")

    def _generate_noise_obs(self, sensor):
        if sensor == "IMU":
            state_with_noise = self.state + self.noise_settings["IMU"].generate(self.state.shape)
            # normalize the orientation
            normalized_ori = th.nn.functional.normalize(state_with_noise[:, 3:], p=2, dim=1)
            state_with_noise = th.cat([
                state_with_noise[:, :3],
                normalized_ori,
                state_with_noise[:, 7:]
            ], dim=1)
            return state_with_noise

    def _create_bbox(self):
        if not self.visual:
            bboxes = [th.tensor([[-5., -10., 0.], [18., 10., 7.]]).to(self.device)]
        # else:
        #     bboxes = []
        #     if self.sceneManager.scenes[0] is None:
        #         self.sceneManager.load_scenes()
        #     for i in range(len(self.sceneManager.scenes)):
        #         bound = None
        #        # use json bound settings as priority
        #         # use habitatsim bound then
        #         if bound is None:
        #             bound = self.sceneManager.scenes[i].get_active_scene_graph().get_root_node().cumulative_bb
        #             bound = habitat_to_std(np.stack([np.array(bound.front_bottom_right), np.array(bound.back_top_left)]), None)[0].to(self.device)
        #         bboxes.append(bound)
            self._bboxes = bboxes
            self._flatten_bboxes = [bbox.flatten() for bbox in bboxes]

    def _create_randomizer(self, random_kwargs: Dict):
        state_random_kwargs = random_kwargs.get("state_generator", {})
        state_generator_class = state_random_kwargs.get("class", UniformStateRandomizer)
        if isinstance(state_generator_class, str):
            state_generator_class = self.state_generator_alias.get(state_generator_class, None)

        stateGenerators,generator_kwargs = [],[]
        kwargs_list = state_random_kwargs.get("kwargs", [])
        if issubclass(state_generator_class, UniformStateRandomizer):
            generator_kwargs = [{
                "position": kwarg.get("position"),
                # "orientation": kwarg.get("orientation", {"mean": [0., 0., 0.], "half": [0., 0., 0.]}),
                # "velocity": kwarg.get("velocity", {"mean": [0., 0., 0.], "half": [0., 0., 0.]}),
                # "angular_velocity": kwarg.get("angular_velocity", {"mean": [0., 0., 0.], "half": [0., 0., 0.]})
            } for kwarg in kwargs_list]
        elif issubclass(state_generator_class, NormalStateRandomizer):
            generator_kwargs = [{
                "position": kwarg.get("position"),
                # "orientation": kwarg.get("orientation", {"mean": [0., 0., 0.], "std": [0., 0., 0.]}),
                # "velocity": kwarg.get("velocity", {"mean": [0., 0., 0.], "std": [0., 0., 0.]}),
                # "angular_velocity": kwarg.get("angular_velocity", {"mean": [0., 0., 0.], "std": [0., 0., 0.]})
            } for kwarg in kwargs_list]
        elif issubclass(state_generator_class, UnionRandomizer):
            # generator_kwargs = [kwarg.get("kwargs") for kwarg in kwargs_list]
            generator_kwargs = kwargs_list
            pass
        else:
            raise ValueError("State generator class is not available.")

        if self.visual:
            stateGenerators = []
            if len(generator_kwargs) == 1:
                for i in range(self.sceneManager.num_scene):
                    for j in range(self.sceneManager.num_agent_per_scene):
                        stateGenerators.append(state_generator_class(
                            device=self.device,
                            is_collision_func=self.sceneManager.get_point_is_collision,
                            scene_id=i,
                            **generator_kwargs[0]
                        )
                        )
            elif len(generator_kwargs) == self.sceneManager.num_scene:
                for i in range(self.sceneManager.num_scene):
                    for j in range(len(generator_kwargs)):
                        stateGenerators.append(state_generator_class(
                            device=self.device,
                            is_collision_func=self.sceneManager.get_point_is_collision,
                            scene_id=i,
                            **generator_kwargs[j]
                        )
                        )
            elif len(generator_kwargs) == self.sceneManager.num_agent:
                for i in range(self.sceneManager.num_agent):
                    stateGenerators.append(state_generator_class(
                        device=self.device,
                        is_collision_func=self.sceneManager.get_point_is_collision,
                        scene_id=i // self.sceneManager.num_agent_per_scene,
                        **generator_kwargs[i]
                    )
                    )

            else:
                raise ValueError("Len of State kwargs is not available.")

            assert len(stateGenerators) == self.sceneManager.num_agent

            for state_generator in stateGenerators:
                state_generator.to(self.device)
                # state_generator.set_seed(self.seed)

        else:
            # not visual
            for _ in range(self.dynamics.num):
                stateGenerators.append(state_generator_class(
                    device=self.device,
                    **generator_kwargs[0]
                )
                )
            # assert len(stateGenerators) == 1

        return stateGenerators

    def _generate_state(self, indices: Optional[List[int]] = None) -> Tuple[Tensor]:
        indices = np.arange(self.dynamics.num) if indices is None else indices
        indices = th.as_tensor([indices], device=self.device) if not hasattr(indices, "__iter__") else indices
        positions, orientations, velocities, angular_velocities = \
            th.empty((len(indices), 3), device=self.device), th.empty((len(indices), 4), device=self.device), \
                th.empty((len(indices), 3), device=self.device), th.empty((len(indices), 3), device=self.device)
        for data_id, index in enumerate(indices):
            positions[data_id], orientations[data_id], velocities[data_id], angular_velocities[data_id] = \
                self.stateGenerators[index].safe_generate(num=1, _eval=self._eval)

        return positions, orientations, velocities, angular_velocities

    def reset(self, state=None) -> Tuple[Tensor, Optional[np.ndarray]]:
        if self.visual:
            if self._scene_iter or self.sceneManager.scenes[0] is None:
                self.sceneManager.load_scenes()
        self.reset_agents(indices=None, state=state)
        return self.state, self.sensor_obs

    def reset_agents(self, indices: Optional[List] = None, state=None) -> Tuple[Tensor, Optional[np.ndarray]]:
        indices = indices if (indices is None or hasattr(indices, "__iter__")) else th.as_tensor([indices], device=self.device)
        motor_speed, thrust, t = None, None, None
        if state is not None:
            if isinstance(state, th.Tensor):
                state = state.to(self.device)
                pos, ori, vel, ori_vel, motor_speed, thrust, t = \
                    state[:,:3].clone().detach(), state[:,3:7].clone().detach(), state[:,7:10].clone().detach(),\
                        state[:,10:13].clone().detach(), state[:,13:17].clone().detach(), state[:,17:21].clone().detach(), state[:,21].clone().detach()
                # state[:, :3], state[:, 3:7], state[:, 7:10], state[:, 10:13], state[:, 13:17], state[:, 17:21]
            else:
                if len(state) == 4:
                    pos, ori, vel, ori_vel = state
                elif len(state) == 6:
                    pos, ori ,vel, ori_vel, motor_speed, thrust = state
                else:
                    raise ValueError("State should be a tuple of 4 or 6 elements.")
        else:
            pos, ori, vel, ori_vel = self._generate_state(indices)
        self.dynamics.reset(pos=pos, ori=ori, vel=vel, ori_vel=ori_vel, motor_omega=motor_speed, thrusts=thrust, t=t, indices=indices)
        if self.visual:
            self.sceneManager.reset_agents(std_positions=pos, std_orientations=ori, indices=indices)
            self.update_observation(indices=indices)
        self.update_collision(indices)

    def update_observation(self, indices=None):
        if indices is None:
            img_obs = self.sceneManager.get_observation()
            # training channel sequence
            for sensor_uuid in self._visual_sensor_list:
                if "depth" in sensor_uuid:
                    self._sensor_obs[sensor_uuid] = \
                        np.expand_dims(np.stack([each_agent_obs[sensor_uuid] for each_agent_obs in img_obs]), 1)
                elif "color" in sensor_uuid:
                    self._sensor_obs[sensor_uuid] = \
                        np.transpose(np.stack([each_agent_obs[sensor_uuid] for each_agent_obs in img_obs])[..., :3], (0, 3, 1, 2))
                elif "semantic" in sensor_uuid:
                    self._sensor_obs[sensor_uuid] = \
                        np.stack([each_agent_obs[sensor_uuid] for each_agent_obs in img_obs])
                else:
                    raise KeyError("Can not find uuid of sensors")
        else:
            img_obs = self.sceneManager.get_observation(indices=indices)
            for each_agent_obs, index in zip(img_obs, indices):
                for sensor_uuid in self._visual_sensor_list:
                    if "depth" in sensor_uuid:
                        self._sensor_obs[sensor_uuid][index, :, :, :] = \
                            np.expand_dims(each_agent_obs[sensor_uuid], 0)
                    elif "color" in sensor_uuid:
                        self._sensor_obs[sensor_uuid][index, :, :, :] = \
                            np.transpose(each_agent_obs[sensor_uuid][..., :3], (2, 0, 1))
                    elif "semantic" in sensor_uuid:
                        self._sensor_obs[sensor_uuid][index, :, :, :] = \
                            each_agent_obs[sensor_uuid]
                    else:
                        raise KeyError("Can not find uuid of sensors")

        self._sensor_obs["IMU"] = self._generate_noise_obs("IMU")

    def update_collision(self, indices: Optional[List[int]] = None):
        if self.visual:
            if indices is None:
                self._collision_point = self.sceneManager.get_collision_point().to(self.device)
            # indices are not None
            else:
                self._collision_point[indices] = self.sceneManager.get_collision_point(indices=indices).to(self.device)
            self._is_out_bounds = self.sceneManager.is_out_bounds

        # not visual
        else:
            if indices is None:
                value, index = th.hstack([
                    self.dynamics.position.clone().detach() - self._bboxes[0][0],
                    self._bboxes[0][1] - self.dynamics.position.clone().detach()]
                ).min(dim=1)
                self._collision_point = self.dynamics.position.clone().detach()
                self._collision_point[th.arange(self.dynamics.num),index%3] = self._flatten_bboxes[0][index]
            else:
                value, index = th.hstack([
                    self.dynamics.position[indices].clone().detach() - self._bboxes[0][0],
                    self._bboxes[0][1] - self.dynamics.position[indices].clone().detach()]
                ).min(dim=1)
                self._collision_point[indices] = self.dynamics.position[indices].clone().detach()
                self._collision_point[indices,index%3] = self._flatten_bboxes[0][index]

            self._is_out_bounds = (self.dynamics.position < self._bboxes[0][0]).any(dim=1) |\
                                  (self.dynamics.position > self._bboxes[0][1]).any(dim=1)

        self._collision_vector = (self._collision_point - self.position)
        self._collision_dis = (self._collision_vector - 0).norm(dim=1)
        self._is_collision = (self._collision_dis < self.uav_radius) | self._is_out_bounds

    def step(self, action):
        self.dynamics.step(action)
        if self.visual:
            self.sceneManager.set_pose(self.dynamics.position, self.dynamics._orientation.toTensor().T)
            self.update_observation()
        self.update_collision()

    def set_seed(self, seed: Union[int, None] = 42):
        seed = self.seed if seed is None else seed
        if self.visual:
            self.sceneManager.seed = seed
        self.dynamics.set_seed(seed)

    def stack(self):
        self._stack_cache = (
            self.position.clone().detach(),
            self.orientation.clone().detach(),
            self.velocity.clone().detach(),
            self.angular_velocity.clone().detach()
        )

    def recover(self):
        self.reset_agents(state=self._stack_cache)

    def detach(self):
        self.dynamics.detach()

    def close(self):
        self.dynamics.close()
        self.sceneManager.close() if self.visual else None

    def render(self, **kwargs):
        if not self.visual:
            raise ValueError("The environment is not visually available.")
        obs = self.sceneManager.render(**kwargs) if self.visual else None
        return obs

    def _find_paths(self, target: th.Tensor, indices=None):
        return self.sceneManager.find_paths(target, indices)

    @property
    def state(self):
        return self.dynamics.state

    @property
    def sensor_obs(self):
        return self._sensor_obs

    @property
    def is_collision(self):
        return self._is_collision

    # @property
    # def closest_obstacle_dis(self):
    #     return self._collision_dis

    @property
    def direction(self):
        return self.dynamics.direction

    @property
    def position(self):
        return self.dynamics.position

    @property
    def orientation(self):
        return self.dynamics.orientation

    @property
    def velocity(self):
        return self.dynamics.velocity

    @property
    def angular_velocity(self):
        return self.dynamics.angular_velocity

    @property
    def t(self):
        return self.dynamics.t

    @property
    def thrusts(self):
        return self.dynamics.thrusts

    @property
    def full_state(self):
        return self.dynamics.full_state

    # @property
    # def acceleration(self):
    #     return self.dynamics.acceleration

    @property
    def collision_point(self):
        return self._collision_point

    @property
    def collision_vector(self):
        return self._collision_vector

    @property
    def collision_dis(self):
        return self._collision_dis

    def eval(self):
        self._eval = True

    def train(self):
        self._eval = False

