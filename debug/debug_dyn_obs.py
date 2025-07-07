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
}]
scene_kwargs = {
    "path": scene_path,
    "obj_settings": {
        "path": "obj",
    },
    "render_settings": {
        "mode": "fix",
        "view": "custom",
        "resolution": [1080, 1920],
        # "position": th.tensor([[-2., 0, 3.5], [2, 0, 2.5]]),
        # "position": th.tensor([[7., 6.8, 5.5], [7, 4.8, 4.5]]),
        "position": th.tensor([[3., 0, 5.5], [7, 0, 4.5]]),
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
    cv.imshow("img", cv.cvtColor(img[0], cv.COLOR_RGBA2BGRA))
    # cv.imshow("obs", np.transpose(obs[0], (1, 2, 0)))
    drone_obs = np.hstack([i[0] for i in obs[:num_agent]])/5
    cv.imshow("obs", cv.cvtColor(drone_obs, cv.COLOR_RGBA2BGRA))
    # print(env.envs.dynamic_object_position)
    # plot_triangle_mesh(env.envs.sceneManager.scenes[0].object_mesh.vertices, faces=env.envs.sceneManager.scenes[0].object_mesh.faces)
    # plot_triangle_mesh(env.envs.sceneManager.scenes[0].scene_mesh.vertices, faces=env.envs.sceneManager.scenes[0].scene_mesh.faces)
    cv.waitKey(100)
    t += 1
