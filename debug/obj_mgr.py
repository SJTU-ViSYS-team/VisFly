#!/usr/bin/env python3

import sys
import os
import torch as th
import numpy as np
import habitat_sim

# Add the correct path to import VisFly modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from VisFly.envs.HoverEnv import HoverEnv2


scene_kwargs = {
    "path":"VisFly/datasets/visfly-beta/configs/garage_empty",
    "sensor_settings":[{
                "sensor_type": habitat_sim.SensorType.DEPTH,
                "uuid": "depth",
                "resolution": [64, 64],
            }],
    "render_settings":{

    },
    "obj_settings":{
        "path":"aa",
        "mode": "static",
        "mode_kwargs": {
            "position": {"mean": [0., 0., 2.2], "half": [1.5, 1.5, 0.5]},
        },
        "generator": "Uniform",
        "generator_kwargs": {

        },

    }

}

env = HoverEnv2(
    scene_kwargs=scene_kwargs,
    num_agent_per_scene=1,
    num_scene=1,
)