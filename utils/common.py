import os
import random
from collections import deque

import numpy as np
from typing import Optional, Tuple, List, Dict
from torch import Tensor
import torch as th
from .maths import Quaternion
from .type import TensorDict


def obs_list2array(obs_dict: List, row: int, column: int):
    obs_indice = 0
    obs_array = []
    for i in range(column):
        obs_row = []
        for j in range(row):
            obs_row.append(obs_dict[obs_indice]["depth"])
            obs_indice += 1
        obs_array.append(np.hstack(obs_row))
    return np.vstack(obs_array)


def depth2rgb(image):
    max_distance = 5.
    image = image / max_distance
    image[image > 1] = 1
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 2:
        image = np.stack([image, image, image, np.full(image.shape, 255, dtype=np.uint8)], axis=-1)
    return image


def rgba2rgb(image):
    if isinstance(image, List):
        return [rgba2rgb(img) for img in image]
    else:
        return image[:, :, :3]


def habitat_to_std(habitat_pos: Optional[np.ndarray] = None, habitat_ori: Optional[np.ndarray] = None, format="enu"):
    """_summary_
        axes transformation, from habitat-sim to std

    Args:
        habitat_pos (_type_): _description_
        habitat_ori (_type_): _description_
        format (str, optional): _description_. Defaults to "enu".

    Returns:
        _type_: _description_
    """
    # habitat_pos, habitat_ori = np.atleast_2d(habitat_pos), np.atleast_2d(habitat_ori)
    assert format in ["enu"]

    if habitat_pos is None:
        std_pos = None
    else:
        # assert habitat_pos.shape[1] == 3
        std_pos = th.as_tensor(
            np.atleast_2d(habitat_pos) @ np.array([[0, -1, 0],
                                                   [0, 0, 1],
                                                   [-1, 0, 0]])
            , dtype=th.float32)
        # if len(habitat_pos.shape) == 1:
        #     std_pos = habitat_pos

    if habitat_ori is None:
        std_ori = None
    else:
        # assert habitat_ori.shape[1] == 4
        std_ori = th.from_numpy(
            np.atleast_2d(habitat_ori) @ np.array(
                [[1, 0, 0, 0],
                 [0, 0, -1, 0],
                 [0, 0, 0, 1],
                 [0, -1, 0, 0]]
            )
        )
    return std_pos, std_ori


def std_to_habitat(std_pos: Optional[Tensor] = None, std_ori: Optional[Tensor] = None, format="enu") \
        -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """_summary_
        axes transformation, from std to habitat-sim

    Args:
        std_pos (_type_): _description_
        std_ori (_type_): _description_
        format (str, optional): _description_. Defaults to "enu".

    Returns:
        _type_: _description_
    """
    assert format in ["enu"]

    # Q = Quaternion(
    #     R.from_euler("ZYX", [-90, 0, 90], degrees=True).as_quat()
    # ).inverse()
    # std_pos_as_quat = [Quaternion(np.r_[std_pos_i, 0]) for std_pos_i in std_pos]
    # hab_pos = np.array([(Q * p * Q.inverse()).imag for p in std_pos_as_quat])
    # std_ori_as_quat = [Quaternion(q) for q in std_ori]
    # hab_ori = np.array(
    #     [(Q * std_ori_as_quat_i).numpy() for std_ori_as_quat_i in std_ori_as_quat]
    # )
    if std_ori is None:
        hab_ori = None
    else:
        hab_ori = std_ori.clone().detach().cpu().numpy() @ np.array(
            [[1, 0, 0, 0],
             [0, 0, 0, -1],
             [0, -1, 0, 0],
             [0, 0, 1, 0]]
        )

    if std_pos is None:
        hab_pos = None
    else:

        if len(std_pos.shape) == 1:
            hab_pos = (std_pos.clone().detach().cpu().unsqueeze(0).numpy() @ np.array([[0, 0, -1],
                                                                                       [-1, 0, 0],
                                                                                       [0, 1, 0]])).squeeze()
        elif std_pos.shape[1] == 3:
            hab_pos = std_pos.clone().detach().cpu().numpy() @ np.array([[0, 0, -1],
                                                                         [-1, 0, 0],
                                                                         [0, 1, 0]])
        else:
            raise ValueError("std_pos shape error")

    return hab_pos, hab_ori


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
    th.manual_seed(seed)

    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    # th.use_deterministic_algorithms(True)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)


def check(returns, r, dones, dim=0):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    ax1.plot(th.stack([th.stack(returns).cpu().T[dim], th.stack(r).cpu().T[dim]]).T)
    ax1.legend(["return", "rewards"])
    ax2 = ax1.twinx()
    ax2.plot(th.stack(dones).cpu().T[dim], 'r-')
    ax2.set_ylabel('dones', color='r')
    ax1.grid()
    plt.show()


def compute_simple_returns(r, dones, next_value, gamma):
    returns = []
    for i in reversed(range(len(r))):
        re = r[i] + gamma * next_value[i] * (~dones[i])
        returns.insert(0, re)

    return th.stack(returns)


# def compute_log_returns(r, dones, next_value, gamma, next_log_prob, ent_coeff):
#     returns = []
#     for i in reversed(range(len(r))):
#         re = r[i] + gamma * (next_value[i]-ent_coeff*next_log_prob[i]) * (~dones[i])
#         returns.insert(0, re)

def compute_monte_carlo_returns(r, dones, next_value, gamma):
    re = next_value[-1]
    returns = []
    for i in reversed(range(len(r))):
        re = r[i] + gamma * re * (~dones[i])
        returns.insert(0, re)

    return th.stack(returns)


def compute_sarsa_returns(r, dones, next_value, gamma, h=36):
    length = len(r)
    returns = 0
    next_value = th.stack(next_value)
    r = th.stack(r + [th.zeros_like(r[0]) for i in range(h)])
    dones = th.stack(dones + [th.ones_like(r[0]) for i in range(h)]).to(th.bool)
    active = ~dones
    accumulative_r = r[h:]
    for i in reversed(range(h)):
        accumulative_r = r[i:(i + length)] + accumulative_r * gamma * active[i:(i + length)]

    next_value_index = (th.arange(length, dtype=th.int32) + h).clamp_max(length - 1)
    next_value_power = 1 + next_value_index - th.arange(length, dtype=th.int32)
    returns = accumulative_r + \
              next_value[next_value_index] * gamma ** (next_value_power.unsqueeze(1) @ th.ones((1, len(r[0])), dtype=th.int32)).to(next_value.device) \
              * dones[next_value_index]
    return returns


def compute_dreamer_returns(r, dones, next_value, gamma, max_h=36, lamda=0.99):
    returns = 0
    for h in range(max_h):
        sarsa_returns = compute_sarsa_returns(r, dones, next_value, gamma, h)
        if h != max_h - 1:
            returns += (1 - lamda) * (lamda ** h) * sarsa_returns
        else:
            returns += (lamda ** (max_h - 1)) * sarsa_returns

    return returns


class RLBuffer:
    rewards = []
    actions = []
    dones = []
    next_observations = []
    observations = []
    # next_value = []
    pre_active = []

    # returns = []

    def clear(self):
        self.rewards = []
        self.actions = []
        self.dones = []
        self.next_observations = []
        self.observations = []
        # self.next_value = []
        self.pre_active = []
        # self.returns = []

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, item):
        buffer = self.__class__()
        self._sub_getitem__(buffer, item)
        return buffer

    def _sub_getitem__(self, buffer, item):
        buffer.rewards = self.rewards[item]
        buffer.actions = self.actions[item]
        buffer.dones = self.dones[item]
        buffer.next_observations = self.next_observations[item]
        buffer.observations = self.observations[item]
        # buffer.next_value = self.next_value[item]
        buffer.pre_active = self.pre_active[item]
        # buffer.returns = self.returns[item]

    def flatten(self):
        buffer = RLBuffer()
        self._flatten(buffer)
        return buffer

    def _flatten(self, buffer):
        buffer.rewards = th.stack(self.rewards).T.flatten()
        buffer.actions = th.stack(self.actions).transpose(0, 1).flatten(end_dim=1)
        buffer.dones = th.stack(self.dones).T.flatten()
        buffer.next_observations = TensorDict.stack(self.next_observations).transpose(0, 1).flatten(end_dim=1)
        buffer.observations = TensorDict.stack(self.observations).transpose(0, 1).flatten(end_dim=1)
        # buffer.next_value = th.stack(self.next_value).T.flatten()
        buffer.pre_active = th.stack(self.pre_active).T.flatten()
        # buffer.returns = self.returns.T.flatten()


class RLBuffer_log(RLBuffer):
    next_prob = []

    def clear(self):
        super().clear()
        self.next_prob = []

    def _sub_getitem__(self, buffer, item):
        super()._sub_getitem__(buffer, item)
        buffer.next_prob = self.next_prob[item]

    def _flatten(self, buffer):
        super()._flatten(buffer)
        buffer.next_prob = th.cat(self.next_prob)


class DataBuffer:
    def __init__(
            self,
            length,
            batch_size,
            observation_space=None,
            action_space=None,
            device="cpu",

    ):
        assert length >= batch_size
        self.length = int(length)
        self.batch_size = int(batch_size)
        self._current_update_index, self._max_available_index = 0, 0
        self.device = th.device(device)
        self._init(observation_space, action_space)

    def _init(self, observation_space, action_space):
        # create a tuple of (name, shape) pairs in observation space
        obs_config = ()
        for key, value in observation_space.items():
            obs_config += (key, (self.length, *value.shape))
        self.observations = TensorDict.empty(*obs_config, device=self.device)
        self.next_observations = TensorDict.empty(*obs_config, device=self.device)
        # same operation for actions
        action_config = (self.length, *action_space.shape)
        self.actions = th.empty(action_config, device=self.device)

        # others
        self.rewards = th.zeros((self.length, 1), device=self.device)
        self.dones = th.zeros((self.length, 1), device=self.device, dtype=th.bool)
        self.pre_active = th.zeros((self.length, 1), device=self.device, dtype=th.bool)

    def append(self, data):
        available_index = th.where(data.pre_active)[0]
        override_index = th.arange(self._current_update_index, self._current_update_index + len(available_index)) % self.length
        self._append(data, available_index, override_index)
        self._current_update_index = (override_index[-1] + 1) % self.length
        if self._max_available_index != self.length - 1:
            self._max_available_index = max(max(override_index), self._max_available_index)

    def _append(self, data, available_index, override_index):
        self.observations[override_index] = data.observations[available_index]
        self.rewards[override_index] = data.rewards[available_index, ...]
        self.actions[override_index] = data.actions[available_index]
        self.next_observations[override_index] = data.next_observations[available_index]
        self.dones[override_index] = data.dones[available_index]

    def sample(self, batch_size=None, env=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        batch_size = min(batch_size, len(self))
        dataset = RLBuffer_log()
        index = random.sample(range(len(self)), batch_size)
        self._sample(dataset, index)
        return dataset

    def _sample(self, dataset, index):
        dataset.observations = self.observations[index]
        dataset.actions = self.actions[index]
        dataset.rewards = self.rewards[index]
        dataset.next_observations = self.next_observations[index]
        dataset.dones = self.dones[index].to(th.int)

    def sample_episode(self, batch_size=None, ):
        datasets = self.sample(batch_size)
        # split the datasets by dones flag
        tensor = datasets.dones
        # Find the indices where the value is True
        true_indices = th.where(tensor)[0] + 1
        # Add the start and end indices
        split_indices = th.cat([th.tensor([0]), true_indices, th.tensor([len(tensor)])])
        # Split the tensor into sections
        episode_dataset = self._sample_episode(datasets, split_indices)
        return episode_dataset

    def _sample_episode(self, dataset, split_indices):
        episode_dataset = dataset.__class__()
        episode_dataset.observations = [dataset.observations[split_indices[i]:split_indices[i + 1]] for i in range(len(split_indices) - 1)]
        episode_dataset.actions = [dataset.actions[split_indices[i]:split_indices[i + 1]] for i in range(len(split_indices) - 1)]
        episode_dataset.rewards = [dataset.rewards[split_indices[i]:split_indices[i + 1]] for i in range(len(split_indices) - 1)]
        episode_dataset.next_observations = [dataset.next_observations[split_indices[i]:split_indices[i + 1]] for i in range(len(split_indices) - 1)]
        episode_dataset.dones = [dataset.dones[split_indices[i]:split_indices[i + 1]] for i in range(len(split_indices) - 1)]

        return episode_dataset

    def __len__(self):
        return self._max_available_index + 1

    @property
    def max_Len(self):
        return self.length

    @property
    def sample_available(self):
        return len(self) >= self.batch_size


class DataBuffer2(DataBuffer):
    def __init__(self, gamma=0.99, length=1e5, batch_size=1e6, observation_space=None, action_space=None, device="cpu"):
        super().__init__(
            length=length,
            batch_size=batch_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device
        )
        self._cache_buffer = RLBuffer()
        self.gamma = gamma

    def _init(self, observation_space, action_space):
        super()._init(observation_space, action_space)
        # self.returns = th.empty(self.length, device=self.device)
        # self.next_value = th.empty(self.length, device=self.device)

    def add(
            self,
            data_dict: Dict[str, Tensor],
    ):
        self._cache_buffer.observations.append(data_dict["observations"])
        self._cache_buffer.rewards.append(data_dict["rewards"])
        self._cache_buffer.actions.append(data_dict["actions"])
        self._cache_buffer.next_observations.append(data_dict["next_observations"])
        self._cache_buffer.dones.append(data_dict["dones"])
        # self._cache_buffer.next_value.append(data_dict["next_value"])
        self._cache_buffer.pre_active.append(data_dict["pre_active"])

    def release(self):
        # self.append(self._cache_buffer.flatten())
        for i in range(len(self._cache_buffer)):
            self.append(self._cache_buffer[i])

        self._cache_buffer.clear()

    # def clear(self):
    #     super().clear()
    #     self.returns = []
    # def _append(self, data, available_index, override_index):
    #     super()._append(data, available_index, override_index)
    # self.returns[override_index] = data.returns[available_index]
    # self.next_value[override_index] = data.returns[available_index]

    # def _sample(self, dataset, index):
    #     super()._sample(dataset, index)
    # dataset.returns = self.returns[index]
    # dataset.next_value = self.next_value[index]

    @property
    def len_cache(self):
        return len(self._cache_buffer)


class DataBuffer3(DataBuffer2):
    def __init__(
            self,
            length,
            batch_size,
            observation_space,
            action_space,
            gamma=0.99,
            device="cpu",
    ):
        super().__init__(
            length=length,
            batch_size=batch_size,
            gamma=gamma,
            observation_space=observation_space,
            action_space=action_space,
            device=device
        )
        self._cache_buffer = RLBuffer_log()

    def _init(self, observation_space, action_space):
        super()._init(observation_space, action_space)
        self.next_prob = th.empty(self.length, device=self.device)

    def add(
            self,
            data_dict,
    ):
        super().add(
            data_dict
        )
        self._cache_buffer.next_prob.append(data_dict["next_prob"])

    def _append(self, data, available_index, override_index):
        super()._append(data, available_index, override_index)
        self.next_prob[override_index] = data.next_prob[available_index]

    def _sample(self, dataset, index):
        super()._sample(dataset, index)
        dataset.next_prob = self.next_prob[index]


from collections import deque


class eq_rollout_buffer:
    def __init__(self, length=100):
        self.reward_buffer = deque(maxlen=length)
        self.success_buffer = deque(maxlen=length)
        self.max_step_buffer = deque(maxlen=length)

        for _ in range(length):
            self.success_buffer.append(False)

    @property
    def rewards(self):
        return sum(self.reward_buffer) / len(self.reward_buffer)

    @property
    def max_step(self):
        return sum(self.max_step_buffer) / len(self.max_step_buffer)

    @property
    def success_rate(self):
        return sum(self.success_buffer) / len(self.success_buffer)
