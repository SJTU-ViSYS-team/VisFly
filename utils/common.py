import copy
import os
import random
from collections import deque

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict

import yaml
from matplotlib import pyplot as plt
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


def depth2color(depth_image, colormap_type='plasma', custom_colormaps=None, max_depth=10):
    """
    Apply different types of colormaps to depth image

    Args:
        depth_image (numpy.ndarray): Input depth image
        colormap_type (str): Type of colormap ('jet', 'viridis', 'plasma', 'custom1', 'custom2')
        custom_colormaps (dict): Dictionary of custom colormaps

    Returns:
        numpy.ndarray: Colored depth image
    """
    # Ensure proper shape
    if len(depth_image.shape) == 3 and depth_image.shape[2] == 1:
        depth_image = depth_image.squeeze()
    else:
        raise ValueError("Invalid depth image shape. Must be HxW or HxWx1")

    # Normalize depth values
    depth_normalized = (depth_image / max_depth).clip(max=1.)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)

    # OpenCV built-in colormaps
    cv2_colormaps = {
        'jet': cv2.COLORMAP_JET,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'plasma': cv2.COLORMAP_PLASMA,
        'magma': cv2.COLORMAP_MAGMA,
        'turbo': cv2.COLORMAP_TURBO,
        'rainbow': cv2.COLORMAP_RAINBOW
    }

    if colormap_type in cv2_colormaps:
        return cv2.applyColorMap(depth_uint8, cv2_colormaps[colormap_type])
    elif custom_colormaps and colormap_type in custom_colormaps:
        # Use matplotlib colormap
        colored = plt.cm.ScalarMappable(cmap=custom_colormaps[colormap_type])
        colored_depth = colored.to_rgba(depth_normalized)[:, :, :3]
        return (colored_depth * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown colormap type: {colormap_type}")

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

def deep_merge(origin, target):
    """
    递归合并字典 `a` 和 `b`，冲突时以 `b` 的值为准。
    若某个键在 `a` 和 `b` 中对应的值均为字典，则递归合并这两个子字典。
    """
    result = copy.deepcopy(origin)
    for key, target_value in target.items():
        origin_value = result.get(key, None)
        if isinstance(origin_value, dict) and isinstance(target_value, dict):
            # 若双方的值均为字典，则递归合并
            result[key] = deep_merge(origin_value, target_value)

        else:
            # 否则直接用 `b` 的值覆盖（深拷贝避免副作用）
            result[key] = copy.deepcopy(target_value)
    return result


def load_yaml_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        if "env" in config:
            config["eval_env"] = deep_merge(origin=config["env"], target=config["eval_env"])

    return config