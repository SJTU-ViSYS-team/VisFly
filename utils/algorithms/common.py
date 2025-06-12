import random
import warnings
from collections import deque
from typing import Union, Optional, Dict, Generator, Any, NamedTuple, List, Tuple
import numpy as np
from gym.vector.utils import spaces
from stable_baselines3.common.buffers import BaseBuffer
import torch as th
from stable_baselines3.common.type_aliases import RolloutBufferSamples, DictRolloutBufferSamples, ReplayBufferSamples, DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from VisFly.utils.type import TensorDict

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

dtype_mapping = {
    np.float32: th.float32,
    np.float64: th.float64,
    np.int32: th.int32,
    np.int64: th.int64,
    np.uint8: th.uint8,
    np.int8: th.int8,
    np.bool_: th.bool,
    # Add more mappings as needed
}


def numpy_to_torch_dtype(np_dtype):
    return dtype_mapping.get(np_dtype.type, None)


def extract_all_paras(model: th.nn.Module) -> Tuple[th.Tensor, th.Tensor]:
    """extract all weight and mer them as a vector, same operation for bias"""
    weights = []
    biases = []
    for name, para in model.named_parameters():
        if "weight" in name:
            weights.append(para.flatten())
        elif "bias" in name:
            biases.append(para.flatten())
    return th.cat(weights), th.cat(biases)

class RolloutBuffer(BaseBuffer):
    """
    A Tensor-type saving version from original buffer in stable-baselines3

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    episode_starts: th.Tensor
    log_probs: th.Tensor
    values: th.Tensor

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = th.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=th.float32)
        self.actions = th.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=th.float32)
        self.rewards = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.returns = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.episode_starts = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.values = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.log_probs = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.advantages = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: th.Tensor) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
            self,
            obs: th.Tensor,
            action: th.Tensor,
            reward: th.Tensor,
            episode_start: th.Tensor,
            value: th.Tensor,
            log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = th.as_tensor(obs)
        self.actions[self.pos] = th.as_tensor(action)
        self.rewards[self.pos] = th.as_tensor(reward)
        self.episode_starts[self.pos] = th.as_tensor(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = th.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
            self,
            batch_inds: th.Tensor,
            env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


class DictRolloutBuffer(RolloutBuffer):
    """
    A Tensor-type saving version from original buffer in stable-baselines3

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to Monte-Carlo advantage estimate when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Dict
    obs_shape: Dict[str, Tuple[int, ...]]  # type: ignore[assignment]
    observations: Dict[str, th.Tensor]  # type: ignore[assignment]

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Dict,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
    ):
        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"

        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = {}
        for key, obs_ithut_shape in self.obs_shape.items():
            self.observations[key] = th.zeros((self.buffer_size, self.n_envs, *obs_ithut_shape), dtype=th.float32)
        self.actions = th.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=th.float32)
        self.rewards = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.returns = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.episode_starts = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.values = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.log_probs = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.advantages = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def add(  # type: ignore[override]
            self,
            obs: Dict[str, th.Tensor],
            action: th.Tensor,
            reward: th.Tensor,
            episode_start: th.Tensor,
            value: th.Tensor,
            log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = th.as_tensor(obs[key])
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = th.as_tensor(action)
        self.rewards[self.pos] = th.as_tensor(reward)
        self.episode_starts[self.pos] = th.as_tensor(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(  # type: ignore[override]
            self,
            batch_size: Optional[int] = None,
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = th.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(  # type: ignore[override]
            self,
            batch_inds: th.Tensor,
            env: Optional[VecNormalize] = None,
    ) -> DictRolloutBufferSamples:
        return DictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
        )


class ReplayBuffer(BaseBuffer):
    """
    A Tensor-type saving version from original buffer in stable-baselines3

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: th.Tensor
    next_observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    dones: th.Tensor
    timeouts: th.Tensor

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = th.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = th.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = th.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.dones = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)

        if psutil is not None:
            total_memory_usage: float = (
                    self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
            self,
            obs: th.Tensor,
            next_obs: th.Tensor,
            action: th.Tensor,
            reward: th.Tensor,
            done: th.Tensor,
            infos: List[Dict[str, Any]],
            state: Optional[th.Tensor] = None,
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = th.as_tensor(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = th.as_tensor(next_obs)
        else:
            self.next_observations[self.pos] = th.as_tensor(next_obs)

        self.actions[self.pos] = th.as_tensor(action)
        self.rewards[self.pos] = th.as_tensor(reward)
        self.dones[self.pos] = th.as_tensor(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = th.as_tensor([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (th.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = th.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: th.Tensor, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = th.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype):
        """
        Cast `th.float64` action datatype to `th.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``th.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == th.float64:
            return th.float32
        return dtype


class DictReplayBuffer(ReplayBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observation_space: spaces.Dict
    obs_shape: Dict[str, Tuple[int, ...]]  # type: ignore[assignment]
    observations: Dict[str, th.Tensor]  # type: ignore[assignment]
    next_observations: Dict[str, th.Tensor]  # type: ignore[assignment]

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Dict,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert not optimize_memory_usage, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: th.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: th.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }

        self.actions = th.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.rewards = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.dones = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage: float = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if not optimize_memory_usage:
                next_obs_nbytes = 0
                for _, obs in self.observations.items():
                    next_obs_nbytes += obs.nbytes
                total_memory_usage += next_obs_nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(  # type: ignore[override]
            self,
            obs: Dict[str, th.Tensor],
            next_obs: Dict[str, th.Tensor],
            action: th.Tensor,
            reward: th.Tensor,
            done: th.Tensor,
            infos: List[Dict[str, Any]],
            state: Optional[th.Tensor] = None,
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = th.as_tensor(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][self.pos] = th.as_tensor(next_obs[key])

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = th.as_tensor(action)
        self.rewards[self.pos] = th.as_tensor(reward)
        self.dones[self.pos] = th.as_tensor(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = th.as_tensor([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(  # type: ignore[override]
            self,
            batch_size: int,
            env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def _get_samples(  # type: ignore[override]
            self,
            batch_inds: th.Tensor,
            env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = th.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}, env)
        next_obs_ = self._normalize_obs(
            {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
        )

        assert isinstance(obs_, dict)
        assert isinstance(next_obs_, dict)
        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
        )


class FullDictReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    states: th.Tensor


class FullDictReplayBuffer(DictReplayBuffer):
    """
    A Tensor-type saving version from original buffer in stable-baselines3
    Including other informations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observation_space: spaces.Dict
    obs_shape: Dict[str, Tuple[int, ...]]  # type: ignore[assignment]
    observations: Dict[str, th.Tensor]  # type: ignore[assignment]
    next_observations: Dict[str, th.Tensor]  # type: ignore[assignment]

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Dict,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.states = th.zeros((self.buffer_size, n_envs, 22), dtype=th.float32)
        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert not optimize_memory_usage, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: th.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=numpy_to_torch_dtype(observation_space[key].dtype))
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: th.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=numpy_to_torch_dtype(observation_space[key].dtype))
            for key, _obs_shape in self.obs_shape.items()
        }

        self.actions = th.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=numpy_to_torch_dtype(self._maybe_cast_dtype(action_space.dtype))
        )
        self.rewards = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.dones = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage: float = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if not optimize_memory_usage:
                next_obs_nbytes = 0
                for _, obs in self.observations.items():
                    next_obs_nbytes += obs.nbytes
                total_memory_usage += next_obs_nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(  # type: ignore[override]
            self,
            obs: Dict[str, th.Tensor],
            next_obs: Dict[str, th.Tensor],
            action: th.Tensor,
            reward: th.Tensor,
            done: th.Tensor,
            infos: List[Dict[str, Any]],
            state: Optional[th.Tensor] = None,
    ) -> None:
        self.states[self.pos] = th.as_tensor(state)

        super().add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=reward,
            done=done,
            infos=infos,
        )

    def sample(  # type: ignore[override]
            self,
            batch_size: int,
            env: Optional[VecNormalize] = None,
    ) -> FullDictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def _get_samples(  # type: ignore[override]
            self,
            batch_inds: th.Tensor,
            env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = th.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}, env)
        next_obs_ = self._normalize_obs(
            {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
        )

        assert isinstance(obs_, dict)
        assert isinstance(next_obs_, dict)
        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return FullDictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
            states=self.to_torch(self.states[batch_inds, env_indices]),
        )

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        # if copy:
        #     return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)


def compute_td_returns(r, done, next_value, episode_done=None, gamma=0.99, lamda=0.95):
    h = len(r)
    num_envs = r[0].shape[0]  # 环境数量
    returns = [th.zeros(num_envs, dtype=th.float32) for _ in range(h)]  # 初始化返回值列表
    episode_done = done if episode_done is None else episode_done

    # 初始化 Ai、Bi 和 lam
    Ai = th.zeros(num_envs, dtype=th.float32, device=r[0].device)
    Bi = th.zeros(num_envs, dtype=th.float32, device=r[0].device)
    lam = th.ones(num_envs, dtype=th.float32, device=r[0].device)
    Bi = next_value[-1] * (~done[-1])
    # 反向循环计算 TD-λ 返回值
    for t in reversed(range(h)):
        active = ~done[t]  # 未终止的环境掩码，1 表示未终止，0 表示已终止
        done_mask = done[t]  # 终止的环境掩码，1 表示已终止，0 表示未终止
        episode_active = ~episode_done[t]
        # 更新 lam，对于已终止的环境，将 lam 重置为 1
        lam = lam * lamda * active + done_mask

        # 更新 Ai
        Ai = active * (
                lamda * gamma * Ai + gamma * next_value[t] + ((1.0 - lam) / (1.0 - lamda)) * r[t]
        )

        # 更新 Bi
        Bi = gamma * (next_value[t] * done_mask * episode_active + Bi * active) + r[t]

        # 计算目标返回值
        returns[t] = (1.0 - lamda) * Ai + lam * Bi

    return returns


class RLBuffer:
    reward = []
    action = []
    done = []
    obs = []
    pre_obs = []
    next_value = []
    pre_active = []

    def clear(self):
        self.reward = []
        self.action = []
        self.done = []
        self.obs = []
        self.pre_obs = []
        self.next_value = []
        self.pre_active = []

    def __len__(self):
        return len(self.reward)


class DataBuffer2:
    def __init__(
            self,
            length,
            batch_size,
    ):
        self.length = int(length)
        self.batch_size = int(batch_size)
        self.reward = []
        self.obs = []
        self.next_obs = []
        self.done = []
        self.action = []
        self._is_buffer_init = False
        self._index_all = range(self.length)
        self._current_update_index = 0

    def append(self, data):
        if not self._is_buffer_init:
            available_index = th.where(data[5])[0]
            self.obs = data[0][available_index]
            self.reward = data[1][available_index]
            self.action = data[2][available_index]
            self.next_obs = data[3][available_index]
            self.done = data[4][available_index]
            # self.value = data[6][available_index]
            self._is_buffer_init = True

        elif len(self.done) == self.length:
            available_index = th.where(data[5])[0]
            new_data_len = len(available_index)
            index = th.arange(self._current_update_index, self._current_update_index + new_data_len) % self.length
            self._current_update_index = (index[-1] + 1) % self.length
            self.obs[index] = data[0][available_index]
            self.reward[index] = data[1][available_index]
            self.action[index] = data[2][available_index]
            self.next_obs[index] = data[3][available_index]
            self.done[index] = data[4][available_index]
            # self.value[index] = data[6][available_index]

        else:
            free_length = self.length - len(self.done)
            available_index = th.where(data[5])[0]
            cut_available_index = available_index[:free_length]
            self.obs.append(data[0][cut_available_index])
            self.reward = th.cat([self.reward, data[1][cut_available_index]])
            self.action = th.cat([self.action, data[2][cut_available_index]])
            self.next_obs.append(data[3][cut_available_index])
            self.done = th.cat([self.done, data[4][cut_available_index]])
            # self.value = th.cat([self.value, data[6][available_index]])

            if free_length < len(available_index):
                self.append((
                    data[0][available_index[free_length:]],
                    data[1][available_index[free_length:]],
                    data[2][available_index[free_length:]],
                    data[3][available_index[free_length:]],
                    data[4][available_index[free_length:]],
                    data[5][available_index[free_length:]],
                    # data[6][available_index[free_length:]],
                ))

    def sample(self):
        assert self.sample_available
        index = random.sample(range(len(self)), self.batch_size)
        return self.obs[index], self.reward[index], self.action[index], self.next_obs[index], self.done[index]

    def __len__(self):
        return len(self.done)

    @property
    def max_Len(self):
        return self.length

    @property
    def sample_available(self):
        return len(self) >= self.batch_size

    def clear(self):
        self.reward = []
        self.obs = []
        self.next_obs = []
        self.done = []
        self.action = []
        self._is_buffer_init = False


class DataBuffer3(DataBuffer2):
    def __init__(self, gamma=0.99, **kwargs):
        super().__init__(**kwargs)
        self._cache_buffer = RLBuffer()
        self.gamma = gamma
        self.returns = None

    # self.replay_buffer.append((pre_obs.clone().detach(), reward.clone().detach(), clipped_actions.clone().detach(), obs.clone().detach(), done, pre_active))

    def add(
            self,
            pre_obs,
            reward,
            action,
            obs,
            done,
            next_value,
            pre_active
    ):
        self._cache_buffer.pre_obs.append(pre_obs)
        self._cache_buffer.reward.append(reward)
        self._cache_buffer.action.append(action)
        self._cache_buffer.obs.append(obs)
        self._cache_buffer.done.append(done)
        self._cache_buffer.next_value.append(next_value)
        self._cache_buffer.pre_active.append(pre_active)

    def release(self):
        returns = compute_td_returns(r=self._cache_buffer.reward,
                                     done=self._cache_buffer.done,
                                     next_value=self._cache_buffer.next_value,
                                     gamma=self.gamma)
        for i in range(len(returns)):
            self.append((
                self._cache_buffer.pre_obs[i],
                self._cache_buffer.reward[i],
                self._cache_buffer.action[i],
                self._cache_buffer.obs[i],
                self._cache_buffer.done[i],
                self._cache_buffer.pre_active[i],
                returns[i]
            ))
        self._cache_buffer.clear()

    def append(self, data):
        if not self._is_buffer_init:
            available_index = th.where(data[5])[0]
            self.obs = data[0][available_index]
            self.reward = data[1][available_index]
            self.action = data[2][available_index]
            self.next_obs = data[3][available_index]
            self.done = data[4][available_index]
            self.returns = data[6][available_index]
            self._is_buffer_init = True

        elif len(self.done) == self.length:
            available_index = th.where(data[5])[0]
            new_data_len = len(available_index)
            index = th.arange(self._current_update_index, self._current_update_index + new_data_len) % self.length
            self._current_update_index = (index[-1] + 1) % self.length
            self.obs[index] = data[0][available_index]
            self.reward[index] = data[1][available_index]
            self.action[index] = data[2][available_index]
            self.next_obs[index] = data[3][available_index]
            self.done[index] = data[4][available_index]
            self.returns[index] = data[6][available_index]

        else:
            free_length = self.length - len(self.done)
            available_index = th.where(data[5])[0]
            cut_available_index = available_index[:free_length]
            self.obs.append(data[0][cut_available_index])
            self.reward = th.cat([self.reward, data[1][cut_available_index]])
            self.action = th.cat([self.action, data[2][cut_available_index]])
            self.next_obs.append(data[3][cut_available_index])
            self.done = th.cat([self.done, data[4][cut_available_index]])
            self.returns = th.cat([self.returns, data[6][available_index]])

            if free_length < len(available_index):
                self.append((
                    data[0][available_index[free_length:]],
                    data[1][available_index[free_length:]],
                    data[2][available_index[free_length:]],
                    data[3][available_index[free_length:]],
                    data[4][available_index[free_length:]],
                    data[5][available_index[free_length:]],
                    data[6][available_index[free_length:]],
                ))

    def sample(self):
        assert self.sample_available
        index = random.sample(range(len(self)), self.batch_size)
        return self.obs[index], self.reward[index], self.action[index], self.next_obs[index], self.done[index], self.returns[index]

    @property
    def len_cache(self):
        return len(self._cache_buffer)

    def clear(self):
        super().clear()
        self.returns = []


class DataBuffer:
    def __init__(
            self,
            length,
            batch_size,
            shuffle=True,
            dtype=tuple,

    ):
        self.length = int(length)
        self.buffer = deque(
            # iterable=dtype,
            maxlen=int(self.length)
        )
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.shuffle = shuffle

    def sample(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        assert self.sample_available
        if self.dtype is tuple:
            data = random.sample(self.buffer, batch_size)
            pre_obs_list = [each_data[0] for each_data in data]
            reward = th.stack([each_data[1] for each_data in data])
            action = th.stack([each_data[2] for each_data in data])
            obs_list = [each_data[3] for each_data in data]
            done = th.stack([each_data[4] for each_data in data])

            pre_obs = {}
            for key in pre_obs_list[0].keys():
                pre_obs[key] = th.stack([each_pre_obs_list[key] for each_pre_obs_list in pre_obs_list])
            obs = {}
            for key in obs_list[0].keys():
                obs[key] = th.stack([each_obs_list[key] for each_obs_list in obs_list])
            return pre_obs, reward, action, obs, done
        else:
            data = None
            raise ValueError("Dtype in replay buffer should be in [tuple]")

    def append(self, data):
        """
            data: s, r, a, s', d, pre_active
        """
        for i in range(len(data[-1])):
            if data[-1][i]:
                self.buffer.append([each_data[i] for each_data in data])

    def __len__(self):
        return len(self.buffer)

    @property
    def max_Len(self):
        return self.length

    @property
    def sample_available(self):
        return len(self) >= self.batch_size


class SimpleRolloutBuffer:
    def __init__(
            self,
            gamma
    ):
        self.gamma = gamma
        self.reward = []
        self.action = []
        self.done = []
        self.obs = []
        self.next_obs = []
        self.value = []
        self.episode_done = []
        self.returns = []

    def clear(self):
        self.reward = []
        self.action = []
        self.done = []
        self.obs = []
        self.next_obs = []
        self.value = []
        self.episode_done = []
        self.returns = []

    def add(self, obs, reward, action, next_obs, done, episode_done, value):
        self.obs.append(obs)
        self.reward.append(reward)
        self.action.append(action)
        self.next_obs.append(next_obs)
        self.done.append(done)
        self.episode_done.append(episode_done)
        self.value.append(value)

    def compute_returns(self):
        self.returns = compute_td_returns(
            r=self.reward,
            done=self.done,
            next_value=self.value,
            episode_done=self.episode_done,
            gamma=0.99
        )
        self.flatten()

    def flatten(self):
        self.reward = th.vstack(self.reward).flatten()
        self.obs = TensorDict.stack(self.obs)
        self.action = th.vstack(self.action)
        self.next_obs = TensorDict.stack(self.next_obs)
        self.done = th.vstack(self.done).flatten()
        self.episode_done = th.vstack(self.episode_done).flatten()
        self.returns = th.vstack(self.returns).flatten()

