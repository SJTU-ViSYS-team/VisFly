from copy import deepcopy

from stable_baselines3.common.vec_env import VecEnv
from .droneEnv import DroneEnvsBase
from typing import Union, Tuple, List, Dict, Optional
from gymnasium import spaces
import numpy as np
from abc import ABC, abstractmethod
import torch as th
from ..utils.type import Uniform
from ..utils.randomization import UniformStateRandomizer
from habitat_sim import SensorType
from ..utils.type import ACTION_TYPE

class DroneGymEnvsBase(VecEnv):
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = False,
            max_episode_steps: int = 1000,
            device: Optional[th.device] = th.device("cpu"),
            dynamics_kwargs=None,
            random_kwargs=None,
            requires_grad: bool = False,
            scene_kwargs: Optional[Dict] = None,
            sensor_kwargs: Optional[List] = None,
            latent_dim=None,
    ):

        super(VecEnv, self).__init__()

        self.envs = DroneEnvsBase(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            # device=device,  # because at least under 1e3 (common useful range) envs, cpu is faster than gpu
            dynamics_kwargs=dynamics_kwargs,
            random_kwargs=random_kwargs,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,

        )

        self.device = device

        self.num_agent = num_agent_per_scene * num_scene
        self.num_scene = num_scene
        self.num_agent_per_scene = num_agent_per_scene
        self.num_envs = self.num_agent

        self.requires_grad = requires_grad

        # key interference of gym env
        state_size = 13 if self.envs.dynamics.is_quat_output else 12

        if not visual:
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)
                }
            )
            for sensor_setting in self.envs.sceneManager.sensor_settings:
                if sensor_setting["sensor_type"] == SensorType.DEPTH:
                    max_depth = np.inf
                    self.observation_space.spaces[sensor_setting["uuid"]] = spaces.Box(
                        low=0, high=max_depth, shape=[1] + sensor_setting["resolution"], dtype=np.float32
                    )
                elif sensor_setting["sensor_type"] == SensorType.COLOR:
                    self.observation_space.spaces[sensor_setting["uuid"]] = spaces.Box(
                        low=0, high=255, shape=[3] + sensor_setting["resolution"], dtype=np.uint8
                    )
                elif sensor_setting["sensor_type"] == SensorType.SEMANTIC:
                    # assert the count of semantic category is less than 256
                    self.observation_space.spaces[sensor_setting["uuid"]] = spaces.Box(
                        low=0, high=255, shape=[1] + sensor_setting["resolution"], dtype=np.uint8
                    )
        if latent_dim is not None:
            self.observation_space["latent"] = spaces.Box(low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32)
            self.latent = th.zeros((self.num_envs, latent_dim))
        else:
            self.latent = None

        if self.envs.dynamics.action_type == ACTION_TYPE.BODYRATE:
            self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        elif self.envs.dynamics.action_type == ACTION_TYPE.THRUST:
            self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        elif self.envs.dynamics.action_type == ACTION_TYPE.VELOCITY:
            self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        else:
            raise ValueError("action_type should be one of ['bodyrate', 'thrust', 'velocity']")

        self._step_count = th.zeros((self.num_agent,), dtype=th.int32)
        self._reward = th.zeros((self.num_agent,))
        self._rewards = th.zeros((self.num_agent,))
        self._action = th.zeros((self.num_agent, 4))

        self._success = th.zeros(self.num_agent, dtype=bool)
        self._done = th.zeros(self.num_agent, dtype=bool)
        self._info = [{"TimeLimit.truncated": False} for _ in range(self.num_agent)]

        self.max_episode_steps = max_episode_steps
        self.auto_examine = not requires_grad

        # necessary for gym compatibility
        self.render_mode = ["None" for _ in range(self.num_agent)]

    def step(self, _action, is_test=False):
        self._action = _action if isinstance(_action, th.Tensor) else th.as_tensor(_action)
        # assert self._action.max() <= self.action_space.high and self._action.min() >= self.action_space.low
        # dot = make_dot(self._action.mean(), "GymEnv")
        # update state and observation and _done
        self.envs.step(self._action)

        observations = self.get_observation()

        self._step_count += 1

        # update success _done
        self._success = self.get_success()

        pre_rewards_debug = self._rewards.clone().detach()
        # update _rewards
        self._reward = self.get_reward()
        self._rewards += self._reward

        # update collision, timeout _done
        test = 1
        self._done = self._done | self._success | self.is_collision | (self._step_count >= self.max_episode_steps)  # | self.get_done()
        # self._done = self._success | (self._step_count >= self.max_episode_steps) | self.get_done()

        # update and record _info: eposide, timeout
        for indice in range(self.num_agent):
            # i don't know why, but whatever this returned info data address should be strictly independent with torch.
            if self._done[indice]:
                if self._success[indice]:
                    self._info[indice]["is_success"] = True
                else:
                    self._info[indice]["is_success"] = False

                self._info[indice]["episode"] = {
                    "r": self._rewards[indice].cpu().clone().detach().numpy(),
                    "l": self._step_count[indice].cpu().clone().detach().numpy(),
                    "t": (self._step_count[indice] * self.envs.dynamics.ctrl_dt).cpu().clone().detach().numpy(),
                }
                if self.requires_grad:
                    self._info[indice]["terminal_observation"] = {
                        key: observations[key][indice].detach() for key in observations.keys()
                    }
                else:
                    self._info[indice]["terminal_observation"] = {
                        key: observations[key][indice] for key in observations.keys()
                    }

                if self._step_count[indice] >= self.max_episode_steps:
                    self._info[indice]["TimeLimit.truncated"] = True

        if not self.requires_grad:
            # model free RL
            _done, _reward, _info = self._done.cpu().clone().numpy(), self._reward.cpu().clone().numpy(), self._info.copy()
            # reset all the dead agents
            if self._done.any() and self.auto_examine and not is_test:
                self.examine()

            return self.get_observation(), _reward, _done, _info
        else:
            # model based RL
            return observations, self._reward, self._done, self._info

    def detach(self):
        self.envs.detach()
        self._rewards = self._rewards.clone().detach()
        self._reward = self._reward.clone().detach()
        self._action = self._action.clone().detach()
        self._step_count = self._step_count.clone().detach()
        self._done = self._done.clone().detach()

    def reset(self):
        self._step_count = th.zeros((self.num_agent,), dtype=th.int32)
        self._reward = th.zeros((self.num_agent,))
        self._rewards = th.zeros((self.num_agent,))
        self._done = th.zeros(self.num_agent, dtype=bool)
        self._info = [{"TimeLimit.truncated": False} for _ in range(self.num_agent)]
        self.envs.reset()
        return self.get_observation()

    def reset_by_id(self, indices=None):
        indices = th.arange(self.num_envs) if indices is None else indices
        self._reward[indices] = 0
        self._rewards[indices] = 0
        self._done[indices] = False
        self._step_count[indices] = 0

        self.envs.reset_agents(indices)
        for indice in indices:
            self._info[indice] = {
                "TimeLimit.truncated": False,
            }

        if self.latent is not None:
            self.latent[indices] = 0

        return self.sensor_obs

    def stack(self):
        self._stack_cache = (self._step_count.clone().detach(),
                             self._reward.clone().detach(),
                             self._rewards.clone().detach(),
                             self._done.clone().detach(),
                             deepcopy(self._info),
                             )
        self.envs.stack()

    def recover(self):
        self._step_count, self._reward, self._rewards, self._done, self._info = \
            self._stack_cache
        self.envs.recover()

    def examine(self):
        if self._done.any():
            self.reset_by_id(th.where(self._done)[0])

    def render(self, render_kwargs={}):
        obs = self.envs.render(render_kwargs)
        return obs

    def get_done(self):
        return th.full((self.num_agent,), False, dtype=bool)

    @abstractmethod
    def get_success(self) -> th.Tensor:
        self._success = np.full((self.num_agent,), False, dtype=bool)

    @abstractmethod
    def get_reward(
            self,
    ) -> Union[np.ndarray, th.Tensor]:
        _rewards = np.empty(self.num_agent)

        return _rewards

    @abstractmethod
    def get_observation(
            self,
            indice=None
    ) -> dict:
        observations = {
            'depth': np.zeros([self.num_agent, 255, 255, 3], dtype=np.uint8),
            'state': np.zeros([self.num_agent, 12], dtype=np.float32),
        }
        return observations

    def close(self):
        self.envs.close()

    @property
    def reward(self):
        return self._reward

    @property
    def sensor_obs(self):
        return self.envs.sensor_obs

    @property
    def state(self):
        return self.envs.state

    @property
    def info(self):
        return self._info

    @property
    def is_collision(self):
        return self.envs.is_collision

    @property
    def done(self):
        return self._done

    @property
    def success(self):
        return self._success

    @property
    def direction(self):
        return self.envs.direction

    @property
    def position(self):
        return self.envs.position

    @property
    def orientation(self):
        return self.envs.orientation

    @property
    def velocity(self):
        return self.envs.velocity

    @property
    def angular_velocity(self):
        return self.envs.angular_velocity

    @property
    def t(self):
        return self.envs.t

    @property
    def visual(self):
        return self.envs.visual

    @property
    def collision_vector(self):
        return self.envs.collision_vector

    @property
    def collision_dis(self):
        return self.envs.collision_dis

    @property
    def collision_point(self):
        return self.envs.collision_point

    def env_is_wrapped(self):
        return False

    def step_async(self):
        raise NotImplementedError('This method is not implemented')

    def step_wait(self):
        raise NotImplementedError('This method is not implemented')

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.
        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        if indices is None:
            return getattr(self, attr_name)

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.
        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise NotImplementedError('This method is not implemented')

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.
        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        raise NotImplementedError('This method is not implemented')

    def to(self, device):
        self.device = device if not isinstance(device, str) else th.device(device)
