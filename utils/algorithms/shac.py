import time
from collections import deque
from typing import Type, Optional, Dict, ClassVar, Any, Union, List

from stable_baselines3.common import logger
import os, sys
from gymnasium import spaces
import random

import numpy as np
import torch as th
from VisFly.utils.policies.td_policies import CnnPolicy, BasePolicy, MultiInputPolicy
# from torch.distributions import
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_schedule_fn, safe_mean, update_learning_rate
from tqdm import tqdm
from stable_baselines3.common.utils import polyak_update, get_parameters_by_name
from VisFly.utils.algorithms.lr_scheduler import  transfer_schedule
from copy import deepcopy
from .common import compute_td_returns, DataBuffer3, SimpleRolloutBuffer
from VisFly.utils.common import set_seed

is_BP_value = True


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)


def check(returns, r, done, dim=0):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    ax1.plot(th.stack([th.stack(returns).cpu().T[dim], th.stack(r).cpu().T[dim]]).T)
    ax1.legend(["return", "reward"])
    ax2 = ax1.twinx()
    ax2.plot(th.stack(done).cpu().T[dim], 'r-')
    ax2.set_ylabel('done', color='r')
    ax1.grid()
    plt.show()


class TemporalDifferBase:
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MultiInputPolicy": MultiInputPolicy,
        "CnnPolicy": CnnPolicy,
    }
    observation_space: spaces.Space
    action_space: spaces.Space
    num_envs: int
    lr_schedule: Schedule

    def __init__(
            self,
            env,
            policy: Union[Type, str],
            policy_kwargs: Optional[Dict] = None,
            learning_rate: Union[float, Schedule] = 1e-3,
            logger_kwargs: Optional[Dict[str, Any]] = None,
            comment: Optional[str] = None,
            save_path: Optional[str] = None,
            dump_step: int = 1e4,
            horizon: float = 32,
            tau: float = 0.005,
            gamma: float = 0.99,
            gradient_steps: int = 5,
            buffer_size: int = int(1e6),
            batch_size: int = int(2e5),
            clip_range_vf: float = 0.1,
            pre_stop: float = 0.1,
            policy_noise: float = 0.,
            device: Optional[str] = "cpu",
            seed: int = 42,
            # **kwargs
    ):
        root = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.save_path = f"{root}/saved" if save_path is None else save_path
        self.device = th.device(device)

        self.env = env
        # self.env.to(self.device)
        self.num_envs = env.num_envs
        self.observation_space: spaces.Dict = env.observation_space
        self.action_space: spaces.Box = env.action_space

        self._dump_step = dump_step
        self.learning_rate = transfer_schedule(learning_rate)
        self.comment = comment
        self.name = "SHAC"

        self._setup_lr_schedule()
        self.logger_kwargs = {} if logger_kwargs is None else logger_kwargs
        self.policy = self._create_policy(policy, policy_kwargs).to(self.device)

        self.H = horizon
        self.tau = tau
        self.gamma = gamma

        self.clip_range_vf = clip_range_vf
        self.pre_stop = pre_stop
        self.gradient_steps = gradient_steps
        self.init_policy_noise = policy_noise
        self.policy_noise = policy_noise

        self._build()
        self.rollout_buffer = SimpleRolloutBuffer(
            gamma=self.gamma
        )

        self._seed = seed
        self._set_seed()

    def _set_seed(self):
        set_seed(self._seed)

    def _build(self):
        self._create_save_path()

        # build a separate evaluation env so that it will not disturb the normal process of training
        try:
            self.eval_env = deepcopy(self.env)
            self.eval_env.reset()
            self.env.reset()
            self.env.requires_grad = True
            # self.eval_env = self.env

        except:
            print("Evaluation env will not be created.")

        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

        self.actor_batch_norm_stats = get_parameters_by_name(self.policy.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.policy.critic, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.policy.critic_target, ["running_"])

    def _create_save_path(self):
        index = 1
        path = f"{self.save_path}/{self.name}_{self.comment}_{index}" if self.comment is not None \
            else f"{self.save_path}/{self.name}_{index}"
        while os.path.exists(path):
            index += 1
            path = f"{self.save_path}/{self.name}_{self.comment}_{index}" if self.comment is not None \
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
    ):
        # assert self.H >= 1, "horizon must be greater than 1"
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

                    fail_cache = th.zeros((self.num_envs,), dtype=th.bool, device=self.device)
                    discount_factor = th.ones((self.num_envs,), dtype=th.float32, device=self.device)
                    episode_done = th.zeros((self.num_envs,), device=self.device, dtype=th.bool)
                    inner_step = 0
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
                        for i in range(len(episode_done)):
                            episode_done[i] = info[i]["episode_done"]

                        reward, done = reward.to(self.device), done.to(self.device)
                        current_step += self.num_envs

                        # compute the temporal difference
                        next_actions, _ = self.policy.actor(obs)
                        next_actions = next_actions.clip(
                            th.as_tensor(self.action_space.low, device=self.device), th.as_tensor(self.action_space.high, device=self.device)
                        )
                        next_values, _ = th.cat(self.policy.critic_target(obs.detach(), next_actions.detach()), dim=1).min(dim=1)

                        # compute the loss
                        # actor_loss += -1 * reward * discount_factor
                        actor_loss = actor_loss - reward * discount_factor
                        done_but_not_episode_end = ((done) | (inner_step == self.H - 1)) & ~episode_done
                        if done_but_not_episode_end.any():
                            actor_loss = actor_loss - next_values * discount_factor * self.gamma * done_but_not_episode_end
                        discount_factor = discount_factor * self.gamma * ~done + done

                        self.rollout_buffer.add(obs=pre_obs.clone().detach(),
                                                reward=reward.clone().detach(),
                                                action=clipped_actions.clone().detach(),
                                                next_obs=obs.clone().detach(),
                                                done=done.clone().detach(),
                                                episode_done=episode_done.clone().detach(),
                                                value=next_values.clone().detach()
                                                )
                    # update
                    actor_loss = (actor_loss).mean()
                    self.policy.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 0.5)
                    # record grad
                    # get_network_statistics(self.actor, self._logger, is_record=pbar.n - previous_step >= self._dump_step)
                    self.policy.actor.optimizer.step()
                    self.rollout_buffer.compute_returns()
                    self.env.detach()

                    # update critic
                    for i in range(self.gradient_steps):
                        values, _ = th.cat(self.policy.critic(self.rollout_buffer.obs, self.rollout_buffer.action), dim=1).min(dim=1)
                        target = self.rollout_buffer.returns
                        critic_loss = th.nn.functional.mse_loss(target, values)
                        self.policy.critic.optimizer.zero_grad()
                        critic_loss.backward()
                        th.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 0.5)
                        self.policy.critic.optimizer.step()

                        polyak_update(params=self.policy.critic.parameters(), target_params=self.policy.critic_target.parameters(), tau=self.tau)
                        polyak_update(params=self.critic_batch_norm_stats, target_params=self.critic_batch_norm_stats_target, tau=1.)

                    self.rollout_buffer.clear()

                    # evaluate
                    if pbar.n - previous_step >= self._dump_step:
                        with th.no_grad():
                            eval_info_id_list = [i for i in range(self.num_envs)]
                            self.eval_env.reset_by_id()
                            obs = self.eval_env.get_observation()
                            while True:
                                actions, _ = self.policy.actor(obs, deterministic=True)
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

    def save(self, path: Optional[str] = None):
        if path is None:
            path = self.policy_save_path
        th.save(self.policy, path + ".pth")
        print(f"Model saved at {path}")

    def predict(self, obs):
        self.policy.eval()
        obs = {key: th.as_tensor(value) for key, value in obs.items()}
        action = self.policy.predict(obs)
        clipped_actions = th.clip(
            action, th.as_tensor(self.action_space.low, device=self.device), th.as_tensor(self.action_space.high, device=self.device)
        )
        with th.no_grad():
            return clipped_actions

    # @staticmethod
    def load(self, path: Optional[str]):
        path += ".pth" if not path.endswith(".pth") else path
        self.policy = th.load(path).to(self.policy.device)
        return self

    @property
    def logger(self):
        return self._logger

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _update_learning_rate(self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        # Log the current learning rate
        self.logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.lr_schedule(self._current_progress_remaining))


    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)


shac = TemporalDifferBase
