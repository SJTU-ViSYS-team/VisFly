import time

import numpy as np
import torch as th
from gymnasium import spaces
from typing import Union, Dict, List, Optional, Type, Any, ClassVar, Tuple

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule
from abc import ABC

from stable_baselines3.common.utils import get_schedule_fn
from tqdm import tqdm
import os, sys
from collections import deque
from stable_baselines3.common import logger
from utils.policies.dl_policies import BaseApgPolicy, ActorPolicy
from utils.algorithms.ppo import ppo


class ApgBase:
    policy_aliases: ClassVar[Dict[str, Type[BaseApgPolicy]]] = {"ActorPolicy": ActorPolicy}
    observation_space: spaces.Space
    action_space: spaces.Space
    num_envs: int
    lr_schedule: Schedule

    def __init__(
            self,
            env,
            policy: Union[Type[BaseApgPolicy], str],
            learning_rate: Union[float, Schedule] = 3e-4,
            policy_kwargs: Optional[Dict] = None,
            logger_kwargs: Optional[Dict[str, Any]] = None,
            commit: Optional[str] = None,
            save_path: Optional[str] = None,
            dump_step: int = 100000,
            device: Optional[str] = "cpu",
            **kwargs
    ):
        root = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.save_path = f"{root}/saved" if save_path is None else save_path
        self.device = th.device(device)

        self.envs = env
        self.num_envs = env.num_envs
        self.observation_space: spaces.Dict = env.observation_space
        self.action_space: spaces.Box = env.action_space

        self._dump_step = dump_step
        self.learning_rate = learning_rate
        self.commit = commit
        self.name = "apg"
        self._create_save_path()

        logger_kwargs = {} if logger_kwargs is None else logger_kwargs
        self.logger_kwargs = logger_kwargs
        self._setup_lr_schedule()
        self.policy = self._create_policy(policy, policy_kwargs).to(self.device)

    def _create_save_path(self):
        index = 1
        path = f"{self.save_path}/{self.name}_{self.commit}_{index}" if self.commit is not None \
            else f"{self.save_path}/{self.name}_{index}"
        while os.path.exists(path):
            index += 1
            path = f"{self.save_path}/{self.name}_{self.commit}_{index}" if self.commit is not None \
                else f"{self.save_path}/{self.name}_{index}"
        self.policy_save_path = path
    def _create_logger(self,
                       format_strings=None,
                       ) -> logger.Logger:
        if format_strings is None:
            format_strings = ["stdout", "tensorboard"]
        l = logger.configure(self.policy_save_path, format_strings)
        return l

    def _create_policy(
            self,
            policy: Type[BasePolicy],
            policy_kwargs: Optional[Dict],
    ):
        policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        if isinstance(policy, str):
            policy_class = self.policy_aliases[policy]
        else:
            policy_class = policy

        policy = policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **policy_kwargs
        )
        return policy

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def log(self, timestep: int):
        raise NotImplementedError

    def learn(
            self,
            total_timesteps: int,
            horizon: int = 1,
    ):
        assert horizon >= 1, "horizon must be greater than 1"
        self.policy.train()
        self._logger = self._create_logger(**self.logger_kwargs)

        buffer_len = 100
        # initialization
        rewards_buffer, len_buffer, success_buffer, len_horizon_buffer = \
            deque(maxlen=buffer_len), deque(maxlen=buffer_len), deque(maxlen=buffer_len), deque(maxlen=buffer_len)

        for _ in range(buffer_len):
            success_buffer.append(False)

        current_step, previous_step, previous_time = 0, 0, 0
        start_time = time.time()
        # self.envs.reset()
        poses = []
        try:
            # for index in iterative:
            with tqdm(total=total_timesteps) as pbar:
                while current_step < total_timesteps:
                    reward_rollout = []

                    self.policy.optimizer.zero_grad()
                    for inner_step in range(horizon):
                        # iteration
                        # action = self.policy(self.envs.get_observation(), deterministic=True)
                        action =self.policy.get_action(self.envs.get_observation())
                        clipped_actions = th.clip(
                            action, th.as_tensor(self.action_space.low, device=self.device), th.as_tensor(self.action_space.high, device=self.device)
                        )
                        self.envs.step(clipped_actions)
                        discount = th.exp(th.as_tensor(-inner_step * 0.02))
                        discount = 1
                        reward_rollout.append(self.envs.get_reward()*discount)
                        current_step += self.num_envs

                        # record the reward and length of the episode
                        for indice in range(self.num_envs):
                            if self.envs.done[indice]:
                                rewards_buffer.append(self.envs.info[indice]["episode"]["r"])
                                len_buffer.append(self.envs.info[indice]["episode"]["l"])
                                success_buffer.append(self.envs.info[indice]["is_success"])

                        if pbar.n- previous_step >= self._dump_step and len(rewards_buffer) > 0:
                            self._logger.record("time/fps", (current_step - previous_step) / (time.time() - previous_time))
                            self._logger.record("time/time", time.time() - start_time)
                            self._logger.record("rollout/ep_rew_mean", sum(rewards_buffer) / len(rewards_buffer))
                            self._logger.record("rollout/ep_len_mean", sum(len_buffer) / len(len_buffer))
                            self._logger.record("rollout/success_rate", sum(success_buffer) / len(success_buffer))
                            self._logger.record("rollout/ep_horizon_mean", sum(len_horizon_buffer) / len(len_horizon_buffer) if len(len_horizon_buffer) >0 else None)
                            self._logger.record("train/total_timesteps", current_step)
                            # self._logger.record("timestep", current_step)
                            self._logger.dump(current_step)
                            previous_time, previous_step = time.time(), current_step

                        if self.envs.done.any():
                            self.envs.reset()
                            break

                    # update
                    len_horizon_buffer.append(inner_step)
                    loss = -1 * th.stack(reward_rollout).mean()
                    loss.backward()
                    self.policy.optimizer.step()
                    pbar.update((inner_step + 1) * self.num_envs)

                    self.envs.detach()

                    # iterative.set_description(f"loss: {loss.item():5.4f}")

        except KeyboardInterrupt:
            save_path = f"{self.policy_save_path}_cache"
            self.save(save_path)

        return self.policy

    def save(self, path: Optional[str] = None):
        if path is None:
            path = self.policy_save_path
        th.save(self.policy, path+".zip")
        print(f"Model saved at {path}")

    def predict(self, obs):
        self.policy.eval()
        obs = {key: th.as_tensor(value) for key, value in obs.items()}
        action = self.policy.get_action(obs)
        clipped_actions = th.clip(
            action, th.as_tensor(self.action_space.low, device=self.device), th.as_tensor(self.action_space.high, device=self.device)
        )
        with th.no_grad():
            return clipped_actions

    def load_from_ppo(self):
        raise NotImplementedError

    @staticmethod
    def load(env, path: Optional[str]):
        self = ApgBase(env=env, policy="ActorPolicy")
        path += ".zip" if not path.endswith(".zip") else path
        self.policy = th.load(path).to(self.policy.device)
        return self

    @property
    def logger(self):
        return self._logger



def debug():
    a = ApgBase(
        env=None,
        policy="ActorPolicy",
        lr_scheduler_class=None,
        policy_kwargs=None,
        lr_schedule_kwargs=None,
        optimizer_class=th.optim.Adam,
        optimizer_kwargs=None,
        logger_kwargs=None,
        commit=None,
        save_path=None,
        policy_name="baseApg",

    )
if __name__ == "__main__":
    debug()
