#!/usr/bin/env python3

import sys
import os
import torch as th
import numpy as np
import cv2

# Add the correct path to import VisFly modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from VisFly.envs.NavigationEnv import NavigationEnv2
from habitat_sim.sensor import SensorType

def test_view_functionality():
    """Test view and rendering functionality"""
    print("=== Testing View and Rendering Functionality ===")
    
    # Configuration
    scene_path = "../datasets/spy_datasets/configs/garage_simple_l_medium"
    sensor_kwargs = [{
        "sensor_type": SensorType.DEPTH,
        "uuid": "depth",
        "resolution": [64, 64],
    }]
    
    random_kwargs = {
        "state_generator": {
            "class": "Uniform",
            "kwargs": [
                {"position": {"mean": [1., 0., 1.5], "half": [2.0, 2.0, 1.0]}},
            ]
        }
}
scene_path = "VisFly/datasets/visfly-beta/configs/garage_empty"
sensor_kwargs = [{
            "sensor_type": SensorType.COLOR,
            "uuid": "depth",
            "resolution": [128, 128],
            "position": [0,0.2,0.],
        }]
scene_kwargs = {
    "path": scene_path,
    "render_settings": {
        "mode": "fix",
        "view": "custom",
        "resolution": [1080, 1920],
        # "position": th.tensor([[6., 6.8, 5.5], [6,4.8,4.5]]),
        "position": th.tensor([[7., 6.8, 5.5], [7, 4.8, 4.5]]),
        "line_width": 6.,

        # "point": th.tensor([[9., 0, 1], [1, 0, 1]]),
        "trajectory": True,
    }
}
num_agent = 4
env = NavigationEnv2(
    visual=True,
    num_scene=1,
    num_agent_per_scene=num_agent,
    random_kwargs=random_kwargs,
    scene_kwargs=scene_kwargs,
    sensor_kwargs=sensor_kwargs,
    dynamics_kwargs={}
)

env.reset()

t= 0
while True:
    a = th.rand((num_agent, 4))
    env.step(a)
    # circile position
    position = th.tensor([[3., 0, 1]]) + th.tensor([[np.cos(t/10), np.sin(t/10), 0]]) * 2
    rotation = Quaternion.from_euler(th.tensor(t/10.), th.tensor(t/10.), th.tensor(t/10)).toTensor().unsqueeze(0)
    # env.envs.sceneManager.set_pose(position=position, rotation=rotation)
    # env.envs.update_observation()
    img = env.render(is_draw_axes=True)
    obs = env.sensor_obs["depth"]
    cv.imshow("img", img[0])
    # cv.imshow("obs", np.transpose(obs[0], (1, 2, 0)))
    cv.imshow("obs", obs[0][0])
    cv.waitKey(100)
    t += 1



