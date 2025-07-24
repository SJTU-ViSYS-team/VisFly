import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from VisFly.envs.NavigationEnv import NavigationEnv2
import cv2 as cv
import torch as th
import time
import numpy as np
from habitat_sim.sensor import SensorType
from VisFly.utils.maths import Quaternion
from VisFly.envs.DynamicEnv import DynEnv
from VisFly.utils.test.mesh_plot import plot_rectangle_mesh, plot_triangle_mesh


def get_batch_mask_centers_torch(mask_batch):
    """PyTorch版本"""
    B, H, W = mask_batch.shape
    centers = []

    for b in range(B):
        mask = mask_batch[b]
        indices = th.nonzero(mask, as_tuple=True)

        if len(indices[0]) == 0:
            centers.append(None)
        else:
            center_y = th.mean(indices[0].float())
            center_x = th.mean(indices[1].float())
            centers.append((center_x.item(), center_y.item()))

    return centers


random_kwargs = {
    "state_generator":
        {
            # "class": "Uniform",
            "class": "TargetUniform",
            "kwargs": [
                {"position": {"mean": [10., 0., 1.5], "half": [1.0, 1.0, 0.3]}},
            ]
        }
}

scene_path = "VisFly/datasets/visfly-beta/configs/scenes/box10_empty"
scene_path = "VisFly/datasets/visfly-beta/configs/scenes/box10_wall"
scene_path = "VisFly/datasets/visfly-beta/configs/scenes/box15_wall_pillar"
# scene_path = "VisFly/datasets/visfly-beta/configs/scenes/garage_simple_l_long"
sensor_kwargs = [{
    "sensor_type": SensorType.DEPTH,
    "uuid": "depth",
    "resolution": [64, 64],
    "position": [0, 0.2, 0.],
},
    {
        "sensor_type": SensorType.SEMANTIC,
        "uuid": "semantic",
        "resolution": [64, 64],
        "position": [0, 0.2, 0.],
    }
]
scene_kwargs = {
    "path": scene_path,
    "obj_settings": {
        "path": "obj2",
    },
    "render_settings": {
        "mode": "fix",
        "view": "custom",
        "resolution": [1080, 1920],
        # "position": th.tensor([[-2., 0, 3.5], [2, 0, 2.5]]),
        # "position": th.tensor([[7., 6.8, 5.5], [7, 4.8, 4.5]]),
        "position": th.tensor([[0., 0, 5.5], [7, 0, 2.5]]),
        "line_width": 6.,

        # "point": th.tensor([[9., 0, 1], [1, 0, 1]]),
        "trajectory": True,
        "collision": True

    }
}
num_agent = 4
num_scene = 1  # 4 agents in each scene
env = DynEnv(
    visual=True,
    num_scene=num_scene,
    num_agent_per_scene=num_agent,
    random_kwargs=random_kwargs,
    scene_kwargs=scene_kwargs,
    sensor_kwargs=sensor_kwargs,
    dynamics_kwargs={},
    tensor_output=True,
)

env.reset()

t = 0
while True:
    # a = th.rand((num_agent*num_scene, 4))
    a = th.zeros((num_agent * num_scene, 4)) + th.tensor([-0.2, 0, 0, 0])
    env.step(a)
    # circile position
    # position = th.tensor([[3., 0, 1]]) + th.tensor([[np.cos(t/10), np.sin(t/10), 0]]) * 2
    # rotation = Quaternion.from_euler(th.tensor(t/10.), th.tensor(t/10.), th.tensor(t/10)).toTensor().unsqueeze(0)
    # env.envs.sceneManager.set_pose(position=position, rotation=rotation)
    # env.envs.update_observation()
    img = env.render(is_draw_axes=True)
    # print(env.position[0])
    obs = env.sensor_obs["depth"]
    semantic = env.sensor_obs["semantic"]
    cv.imshow("img", cv.cvtColor(img[0], cv.COLOR_RGBA2BGRA))
    # cv.imshow("obs", np.transpose(obs[0], (1, 2, 0)))
    drone_obs = np.hstack([i[0] for i in obs[:num_agent]])/5
    drone_seg = np.hstack([i[0] for i in semantic[:num_agent]])
    cv.imshow("depth", cv.cvtColor(drone_obs, cv.COLOR_RGBA2BGRA))
    drone_seg = np.where(drone_seg==5, drone_seg, 0)
    mask = drone_seg==5
    centers = get_batch_mask_centers_torch(th.tensor(semantic==5).squeeze())
    drone_seg_vis = drone_seg.copy().astype(np.float32)

    # 在每个agent的图像区域画center点
    img_width = 64  # 单个图像宽度
    for i, center in enumerate(centers):
        if center is not None:
            # 计算在拼接图像中的实际坐标
            offset_x = i * img_width
            actual_x = int(center[0] + offset_x)
            actual_y = int(center[1])

            # 画圆心
            cv.circle(drone_seg_vis, (actual_x, actual_y), 3, 1.0, -1)  # 白色填充圆
            cv.circle(drone_seg_vis, (actual_x, actual_y), 5, 0.5, 2)  # 灰色外圈

            print(f"Agent {i} - Center: ({center[0]:.1f}, {center[1]:.1f})")
    cv.imshow("semantic", cv.cvtColor(drone_seg_vis / 9, cv.COLOR_RGBA2BGRA))
    # cv.imshow("semantic", cv.cvtColor(drone_seg.astype(np.float32)/9, cv.COLOR_RGBA2BGRA))
    # print(env.envs.dynamic_object_position)
    # plot_triangle_mesh(env.envs.sceneManager.scenes[0].object_mesh.vertices, faces=env.envs.sceneManager.scenes[0].object_mesh.faces)
    # plot_triangle_mesh(env.envs.sceneManager.scenes[0].scene_mesh.vertices, faces=env.envs.sceneManager.scenes[0].scene_mesh.faces)
    cv.waitKey(1)
    t += 1
