#!/usr/bin/env python3

import sys
import os
import torch as th
import numpy as np

# Add the correct path to import VisFly modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from VisFly.envs.NavigationEnv import NavigationEnv2
from habitat_sim.sensor import SensorType

def test_scene_reset():
    """Test scene reset functionality"""
    print("=== Testing Scene Reset Functionality ===")
    
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
    sensor_kwargs=sensor_kwargs
)

env.reset()
env.reset_env_by_id(0)
