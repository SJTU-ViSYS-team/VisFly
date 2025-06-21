import warnings
from copy import deepcopy

from stable_baselines3.common.vec_env import VecEnv
from .droneEnv import DroneEnvsBase
from typing import Union, Tuple, List, Dict, Optional
from gymnasium import spaces
import numpy as np
from abc import ABC, abstractmethod
import torch as th
from habitat_sim import SensorType
from ...utils.type import ACTION_TYPE, TensorDict


class DroneGymEnvsBase(VecEnv):
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = False,
            max_episode_steps: int = 1000,
            device: Optional[th.device] = th.device("cpu"),
            dynamics_kwargs={},
            random_kwargs={},
            requires_grad: bool = False,
            scene_kwargs: Optional[Dict] = {},
            sensor_kwargs: Optional[List] = [],
            tensor_output: bool = False,
            is_train: bool = False,
    ):

        super(VecEnv, self).__init__()

        # raise Warning if device is cuda while num_envs is less than 1e3
        device = th.device(device)
        if num_agent_per_scene * num_scene < 1e3 and (device.type == 'cuda'):
            _env_device = th.device("cpu")
            warnings.warn("The number of envs is less than 1e3, cpu is faster than gpu. To make training faster, we have set device to cpu.")
        else:
            _env_device = device

        # sensor_kwargs = [{
        #     "sensor_type": SensorType.DEPTH,
        #     "uuid": "depth",
        #     "resolution": [64, 64],
        # }] if sensor_kwargs is None else sensor_kwargs
        # random_kwargs = {
        #     "state_generator":
        #         {
        #             "class": "Uniform",
        #             "kwargs": [
        #                 {"position": {"mean": [1., 0., 1.5], "half": [0.0, 2., 1.]}},
        #             ]
        #         }
        # } if random_kwargs is None else random_kwargs
        # dynamics_kwargs = {
        #     "dt": 0.02,
        #     "ctrl_dt": 0.02,
        #     "action_type": "bodyrate",
        #     "ctrl_delay": True,
        # } if dynamics_kwargs is None else dynamics_kwargs
        # scene_kwargs = {
        #     "path": "VisFly/datasets/spy_datasets/configs/garage_empty"
        # } if scene_kwargs is None else scene_kwargs

        self.envs = DroneEnvsBase(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            device=_env_device,  # because at least under 1e3 (common useful range) envs, cpu is faster than gpu
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
        self.max_sense_radius = 10

        self.tensor_output = tensor_output
        self.is_train = is_train

        # key interference of gym env
        state_size = 3 + 3+ 3 +(3 if self.envs.dynamics.angular_output_type == "euler" else (4 if self.envs.dynamics.angular_output_type == "quaternion" else 6))

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
        # if latent_dim is not None:
        #     # self.observation_space["latent"] = spaces.Box(low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32)
        #     self.latent = th.zeros((self.num_envs, latent_dim), device=self.device)
        self.deter = None
        self.stoch = None

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
        self._observations = TensorDict({})

        self._success = th.zeros(self.num_agent, dtype=bool)
        self._failure = th.zeros(self.num_agent, dtype=bool)
        self._episode_done = th.zeros(self.num_agent, dtype=bool)
        self._done = th.zeros(self.num_agent, dtype=bool)
        self._info = [{"TimeLimit.truncated": False} for _ in range(self.num_agent)]

        self._indiv_rewards = None
        self.max_episode_steps = max_episode_steps

        # necessary for gym compatibility
        self.render_mode = ["None" for _ in range(self.num_agent)]

    def step(self, _action, is_test=False, latent_func=None):
        self._action = _action if isinstance(_action, th.Tensor) else th.as_tensor(_action)
        assert self._action.max() <= 1 and self._action.min() >= -1
        # update state and observation and _done
        self.envs.step(self._action)
        self.get_full_observation()
        if latent_func is not None:
            self.update_latent(latent_func=latent_func)

        self._step_count += 1

        # update success _done
        self._success = self.get_success()
        self._failure = self.get_failure()

        # update _rewards
        if self._indiv_reward is None:
            self._reward = self.get_reward()
        else:
            self._indiv_reward = self.get_reward()
            assert isinstance(self._indiv_reward, dict) and "reward" in self._indiv_reward.keys()
            self._reward = self._indiv_reward["reward"]
            for key in self._indiv_reward.keys():
                # Ensure device compatibility by moving reward tensor to same device as accumulated rewards
                reward_tensor = self._indiv_reward[key]
                if hasattr(reward_tensor, 'to'):
                    reward_tensor = reward_tensor.to(self._indiv_rewards[key].device)
                self._indiv_rewards[key] += reward_tensor
        
        # Ensure _reward is on the same device as _rewards before addition
        if hasattr(self._reward, 'to'):
            self._reward = self._reward.to(self._rewards.device)
        self._rewards += self._reward

        # update collision, timeout _done
        # Ensure all boolean tensors are on the same device before logical operations
        success = self._success.to(self._episode_done.device) if hasattr(self._success, 'to') else self._success
        failure = self._failure.to(self._episode_done.device) if hasattr(self._failure, 'to') else self._failure
        collision = self.is_collision.to(self._episode_done.device) if hasattr(self.is_collision, 'to') else self.is_collision
        
        self._episode_done = self._episode_done | success | failure | collision
        # self._episode_done = self._episode_done | self._success | self._failure

        self._done = self._episode_done | (self._step_count >= self.max_episode_steps)

        # update and record _info: episode, timeout
        # full_state = self.full_state
        for indice in range(self.num_agent):
            # self._info[indice]["state"]= full_state[indice].cpu().clone().detach().numpy()
            # i don't know why, but whatever this returned info data address should be strictly independent with torch.
            if self._done[indice]:
                self._info[indice] = self.collect_info(indice, self._observations)
            else:
                # Ensure episode_done is always available for all agents
                self._info[indice]["episode_done"] = self._episode_done[indice].clone().detach()

        # return and auto-reset
        _done, _reward, _info = self._done.clone(), self._reward.clone(), self._info.copy()
        # _episode_done = self._episode_done.clone()
        # reset all the dead agents
        if self._done.any() and not is_test:
            self.examine()
        if self.requires_grad:
            # analytical gradient RL
            return self._observations, _reward, _done, _info
        else:
            if self.tensor_output:
                return self._observations.detach(), _reward.detach(), _done, _info
            else:
                return self._observations, _reward.cpu().numpy(), _done.cpu().numpy(), _info

    @th.no_grad()
    def update_latent(self, latent_func):
        next_stoch_post, next_deter = latent_func(
                    action=self._action,
                    stoch=self.stoch,
                    deter=self.deter,
                    deterministic=False,
                    next_observation=self._observations,
                    # return_prior=True,
                )
        self.deter = next_deter.to(self.device)
        self.stoch = next_stoch_post.to(self.device)
        self._observations["deter"] = self.deter.detach().cpu()
        self._observations["stoch"] = self.stoch.detach().cpu()

    def collect_info(self, indice, observations):
        _info = {}
        if self._success[indice]:
            _info["is_success"] = True
        else:
            _info["is_success"] = False

        # Add episode_done field that ABPT algorithm expects
        _info["episode_done"] = self._episode_done[indice].clone().detach()

        _info["episode"] = {
            "r": self._rewards[indice].cpu().clone().detach().numpy(),
            "l": self._step_count[indice].cpu().clone().detach().numpy(),
            "t": (self._step_count[indice] * self.envs.dynamics.ctrl_dt).cpu().clone().detach().numpy(),
        }
        if self.requires_grad:
            _info["terminal_observation"] = {
                key: observations[key][indice].detach() if hasattr(observations[key][indice], 'detach') else observations[key][indice] for key in observations.keys()
            }
        else:
            _info["terminal_observation"] = {
                key: observations[key][indice] for key in observations.keys()
            }

        if self._step_count[indice] >= self.max_episode_steps:
            _info["TimeLimit.truncated"] = True
        _info["episode"]["extra"] = {}
        _info["episode_done"] = self._episode_done[indice].clone().detach()

        if self._indiv_rewards is not None:
            for key in self._indiv_rewards.keys():
                _info["episode"]["extra"][key] = self._indiv_rewards[key][indice].clone().detach()

        return _info

    def initialize_latent(self, deter_dim, stoch_dim, latent_reset_func=None, latent_func=None, world=None):
        self.deter = th.zeros((self.num_agent, deter_dim), device=self.device)
        self.stoch = th.zeros((self.num_agent, stoch_dim), device=self.device)
        self.observation_space["deter"] = spaces.Box(low=-np.inf, high=np.inf, shape=(deter_dim,), dtype=np.float32)
        self.observation_space["stoch"] = spaces.Box(low=-np.inf, high=np.inf, shape=(stoch_dim,), dtype=np.float32)
        if latent_reset_func:
            setattr(self, "latent_reset_func", latent_reset_func)
        if latent_func:
            setattr(self, "latent_func", latent_func)
        if world:
            setattr(self, "world", world)

    # def set_latent(self, deter, stoch):
    #     self.deter = deter
    #     self.stoch = stoch

    def detach(self):
        self.envs.detach()
        self.simple_detach()

    def simple_detach(self):
        self._rewards = self._rewards.clone().detach()
        self._reward = self._reward.clone().detach()
        self._action = self._action.clone().detach()
        self._step_count = self._step_count.clone().detach()
        self._done = self._done.clone().detach()
        if hasattr(self, 'latent'):
            self.latent = self.latent.clone().detach()

    def reset(self, state=None, obs=None, is_test=False):
        self.envs.reset()

        if isinstance(self.get_reward(), dict):
            self._indiv_reward: dict = self.get_reward()
            self._indiv_rewards = {key: th.zeros((self.num_agent,), device=self.device) for key in self._indiv_reward.keys()}
            self._indiv_reward = {key: th.zeros((self.num_agent,), device=self.device) for key in self._indiv_reward.keys()}
        elif isinstance(self.get_reward(), th.Tensor):
            self._indiv_rewards = None
            self._indiv_reward = None

        else:
            raise ValueError("get_reward should return a dict or a tensor, but got {}".format(type(self.get_reward())))

        self.get_full_observation()
        self._reset_attr()
        self.get_full_observation()

        return self._observations
        #
        # if not self.tensor_output:
        #     return self.get_full_observation()
        # else:
        #     return self.get_full_observation()

    def reset_env_by_id(self, scene_indices=None):
        assert not isinstance(scene_indices, bool)
        scene_indices = scene_indices if scene_indices is not None else th.arange(self.num_scene).tolist()
        scene_indices = [scene_indices] if not hasattr(scene_indices, "__iter__") else scene_indices
        self.envs.reset_scenes(scene_indices)
        agent_indices = ((np.tile(np.arange(self.num_agent_per_scene), (len(scene_indices), 1))
                          + (scene_indices * self.num_agent_per_scene)).reshape(-1, 1)).flatten()
        self._reset_attr(indices=agent_indices)
        return self.get_full_observation(agent_indices)

    def reset_agent_by_id(self, agent_indices=None, state=None, reset_obs=None):
        assert ~(state is None and reset_obs is None) or (state is not None and reset_obs is not None)
        assert not isinstance(agent_indices, bool)
        self._reset_attr(indices=agent_indices)
        self.envs.reset_agents(agent_indices, state=state)
        return self.get_full_observation(agent_indices)

    def _format_obs(self, obs):
        if not self.tensor_output:
            return obs.detach().cpu().numpy()
        else:
            return obs

    def _reset_attr(self, indices=None):
        """
        Resets the internal state of the environments.
        :param indices: indices of envs to reset. If None, resets all envs.
        """
        # All internal state tensors should be on the simulation device (CPU)
        sim_device = self.envs.device if hasattr(self.envs, 'device') else self.device

        if indices is None:
            self._reward = th.zeros((self.num_agent,), device=self.device)
            self._rewards = th.zeros((self.num_agent,), device=self.device)
            self._done = th.zeros(self.num_agent, dtype=bool, device=self.device)
            self._episode_done = th.zeros(self.num_agent, dtype=bool, device=self.device)
            self._step_count = th.zeros((self.num_agent,), dtype=th.int32, device=self.device)
            if self.deter is not None:
                if hasattr(self, "world"):
                    latent = self.world.sequence_model.initial(self.num_agent)
                    # next_deter, next_stoch_post = latent["deter"], latent["stoch"]
                    next_stoch, next_deter = self.world.sequence_model(
                        action=th.zeros((self.num_agent, 4), device=self.world.sequence_model.device),
                        stoch=latent["stoch"],
                        deter=latent["deter"],
                        deterministic=False
                    )
                    next_stoch_post = self.world.encoder(observation=self._observations,
                                                                 deter=next_deter,
                                                                 deterministic=False)
                    self.deter, self.stoch = next_deter.to(self.device), next_stoch_post.to(self.device)
                else:
                    self.deter = th.zeros_like(self.deter, device=self.device)
                    self.stoch = th.zeros_like(self.stoch, device=self.device)
            if self._indiv_rewards is not None:
                for key in self._indiv_rewards.keys():
                    self._indiv_rewards[key] = th.zeros((self.num_agent,), device=self.device)
                    self._indiv_reward[key] = th.zeros((self.num_agent,), device=self.device)

        else:
            # These are in-place modifications, device is already correct.
            self._step_count[indices] = 0
            self._reward[indices] = 0.
            self._rewards[indices] = 0.
            self._action[indices] = 0.
            self._success[indices] = False
            self._failure[indices] = False
            self._episode_done[indices] = False
            self._done[indices] = False
            for i in indices:
                self._info[i] = {"TimeLimit.truncated": False}

        if self._indiv_rewards is not None:
            for key in self._indiv_rewards.keys():
                if indices is None:
                    # This was already fixed, but ensure it uses sim_device
                    self._indiv_rewards[key] = th.zeros((self.num_agent,), device=sim_device)
                else:
                    self._indiv_rewards[key][indices] = 0
                    self._indiv_reward[key][indices] = 0

        indices = range(self.num_agent) if indices is None else indices
        for indice in indices:
            self._info[indice] = {
                "TimeLimit.truncated": False,
            }

    # def stack(self):
    #     self._stack_cache = (self._step_count.clone().detach(),
    #                          self._reward.clone().detach(),
    #                          self._rewards.clone().detach(),
    #                          self._done.clone().detach(),
    #                          deepcopy(self._info),
    #                          )
    #     self.envs.stack()

    def recover(self):
        self._step_count, self._reward, self._rewards, self._done, self._info = \
            self._stack_cache
        self.envs.recover()

    def examine(self):
        if self._done.any():
            # Ensure indices are on CPU for compatibility with dynamics system
            done_indices = th.where(self._done)[0].cpu()
            self.reset_agent_by_id(done_indices)
        return self.get_full_observation()

    def render(self, **kwargs):
        obs = self.envs.render(**kwargs)
        return obs

    def get_done(self):
        return th.full((self.num_agent,), False, dtype=bool)

    @abstractmethod
    def get_success(self) -> th.Tensor:
        _success = th.full((self.num_agent,), False, dtype=bool)
        return _success

    def get_failure(self) -> th.Tensor:
        _failure = th.full((self.num_agent,), False, dtype=bool)
        return _failure

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

    def get_full_observation(self, indice=None):
        obs = self.get_observation()
        assert isinstance(obs, TensorDict)

        if self.deter is not None:
            obs.update({"deter": self.deter})
            obs.update({"stoch": self.stoch})

        self._observations = self._format_obs(obs.as_tensor(device=self.device))
        if indice is None:
            return self._observations
        else:
            return self._observations[indice]

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
    def episode_done(self):
        return self._episode_done

    @property
    def success(self):
        return self._success

    @property
    def failure(self):
        return self._failure

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

    @property
    def full_state(self):
        return self.envs.full_state

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

    def eval(self):
        self.envs.eval()

    def __len__(self):
        return self.num_envs