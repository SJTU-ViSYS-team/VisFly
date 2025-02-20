# this function is to generate scenes based on square stage.

import numpy as np
import torch as th
scene_example = {
    "stage_instance": {
        "template_name": "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    },
    # "collision_asset":"data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    "default_lighting": "data/scene_datasets/habitat-test-scenes/default_lighting.glb",
    "object_instances": [
        {
            "template_name": "data/objects/example_object.glb",
            "translation": [1.0,
                            0.0,
                            0.0],
            "rotation": [0.0,
                         0.0,
                         0.0,
                         1.0],
            "uniform_scale": 1.0,
        },
    ],
    "articulated_object_instances": [
        {
            "template_name": "fridge",
            "translation_origin": "COM",
            "fixed_base": True,
            "translation": [
                -2.1782121658325195,
                0.9755649566650391,
                3.2299728393554688
            ],
            "rotation": [
                1,
                0,
                0,
                0
            ],
            "motion_type": "DYNAMIC"
        },
    ],
    # "navmesh_instance": "empty_stage_navmesh",
    "default_lighting": "",
    "user_custom": {
        "bound": [
            [1, 1, 1],
            [10, 10, 10]
        ]
    }
}