from .shac import *
import time
from collections import deque
from typing import Type, Optional, Dict, ClassVar, Any, Union

from stable_baselines3.common import logger
import os, sys
from gymnasium import spaces
import random

import numpy as np
import torch as th
from VisFly.utils.policies.td_policies import CnnPolicy, BasePolicy, MultiInputPolicy
from stable_baselines3.sac.policies import MultiInputPolicy as SACMultiInputPolicy
# from torch.distributions import
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_schedule_fn
from tqdm import tqdm
from stable_baselines3.common.utils import polyak_update, get_parameters_by_name
from VisFly.utils.algorithms.lr_scheduler import transfer_schedule
from VisFly.utils.test.debug import get_network_statistics, check_none_parameters
from copy import deepcopy
from stable_baselines3.common.utils import safe_mean

is_BP_value = False


class BPTT(shac):
    def __init__(
            self,
            env,
            policy,
            policy_kwargs=None,
            learning_rate=1e-3,
            logger_kwargs=None,
            comment=None,
            save_path=None,
            dump_step: int = 1e4,
            horizon: float = 32,
            tau: float = 0.005,
            gamma: float = 0.99,
            buffer_size: int = int(1e6),
            batch_size: int = int(2e5),
            clip_range_vf: float = 0.1,
            pre_stop: float = 0.1,
            policy_noise: float = 0.,
            device="cpu",
            seed: int = 42,
            # **kwargs
    ):

        super().__init__(
            env=env,
            policy=policy,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            logger_kwargs=logger_kwargs,
            comment=comment,
            save_path=save_path,
            dump_step=dump_step,
            horizon=horizon,
            tau=tau,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
            clip_range_vf=clip_range_vf,
            pre_stop=pre_stop,
            policy_noise=policy_noise,
            device=device,
            seed=seed,
        )

    def _build(self):
        self.name = "BPTT"
        super()._build()

    def learn(
            self,
            total_timesteps: int,
    ):
        assert self.H >= 1, "horizon must be greater than 1"
        self.policy.train()
        self._logger = self._create_logger(**self.logger_kwargs)

        eq_buffer_len = 100
        # initialization
        eq_rewards_buffer, eq_len_buffer, eq_success_buffer, eq_info_buffer = \
            deque(maxlen=eq_buffer_len), deque(maxlen=eq_buffer_len), deque(maxlen=eq_buffer_len), deque(maxlen=eq_buffer_len)

        for _ in range(eq_buffer_len):
            eq_success_buffer.append(False)

        current_step, previous_step, previous_time = 0, 0, 0
        try:
            with (tqdm(total=total_timesteps) as pbar):
                while current_step < total_timesteps:
                    self._update_current_progress_remaining(num_timesteps=current_step, total_timesteps=total_timesteps)
                    optimizers = [self.actor.optimizer, self.critic.optimizer]
                    # Update learning rate according to lr schedule
                    self._update_learning_rate(optimizers)

                    actor_loss, critic_loss = 0., 0.  # th.tensor(0, device=self.device), th.tensor(0, device=self.device)
                    obs = self.env.get_observation()

                    discount_factor = th.ones((self.num_envs,), dtype=th.float32, device=self.device)

                    for inner_step in range(self.H):
                        # dream a horizon of experience
                        obs = self.env.get_observation()
                        pre_obs = obs
                        # iteration
                        actions, _, h = self.policy.actor.action_log_prob(obs)
                        clipped_actions = th.clip(
                            actions, th.as_tensor(self.action_space.low, device=self.device), th.as_tensor(self.action_space.high, device=self.device)
                        )

                        # step
                        obs, reward, done, info = self.env.step(clipped_actions)
                        reward, done = reward.to(self.device), done.to(self.device).to(th.bool)
                        current_step += self.num_envs

                        # compute the loss
                        actor_loss += -1 * reward * discount_factor
                        discount_factor = discount_factor * self.gamma * ~done + done

                    # update
                    actor_loss = (actor_loss).mean()
                    self.policy.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 0.5)
                    # record grad
                    # get_network_statistics(self.actor, self._logger, is_record=pbar.n - previous_step >= self._dump_step)
                    self.policy.actor.optimizer.step()
                    self.env.detach()

                    # evaluate
                    if pbar.n - previous_step >= self._dump_step:
                        with th.no_grad():
                            eval_info_id_list = [i for i in range(self.num_envs)]
                            self.eval_env.reset_agent_by_id()
                            obs = self.eval_env.get_observation()
                            while True:
                                actions, _ = self.policy.actor(obs)
                                clipped_actions = th.clip(
                                    actions, th.as_tensor(self.action_space.low, device=self.device), th.as_tensor(self.action_space.high, device=self.device)
                                )
                                obs, reward, done, info = self.eval_env.step(clipped_actions, is_test=True)
                                for index in reversed(eval_info_id_list):
                                    if done[index]:
                                        eval_info_id_list.remove(index)
                                        eq_rewards_buffer.append(info[index]["episode"]["r"])
                                        eq_len_buffer.append(info[index]["episode"]["l"])
                                        eq_success_buffer.append(info[index]["is_success"])
                                        eq_info_buffer.append(info[index]["episode"])
                                if done.all():
                                    break

                    if pbar.n - previous_step >= self._dump_step and len(eq_rewards_buffer) > 0:
                        self._logger.record("time/fps", (current_step - previous_step) / (time.time() - previous_time))
                        self._logger.record("rollout/ep_rew_mean", sum(eq_rewards_buffer) / len(eq_rewards_buffer))
                        self._logger.record("rollout/ep_len_mean", sum(eq_len_buffer) / len(eq_len_buffer))
                        self._logger.record("rollout/success_rate", sum(eq_success_buffer) / len(eq_success_buffer))
                        self._logger.record("train/actor_loss", actor_loss.item())
                        self._logger.record("train/critic_loss", critic_loss.item() if isinstance(critic_loss, th.Tensor) else critic_loss)
                        if len(eq_info_buffer[0]["extra"]) >= 0:
                            for key in eq_info_buffer[0]["extra"].keys():
                                self.logger.record(
                                    f"rollout/ep_{key}_mean",
                                    safe_mean(
                                        [ep_info["extra"][key] for ep_info in eq_info_buffer]
                                    ),
                                )
                        self._logger.dump(current_step)
                        previous_time, previous_step = time.time(), current_step
                    pbar.update((inner_step + 1) * self.num_envs)

        except KeyboardInterrupt:
            pass

        return self.policy
