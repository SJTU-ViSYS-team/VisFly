from VisFly.envs.HoverEnv import HoverEnv2
import time
import torch as th
from VisFly.envs.DynamicEnv import DynEnv
from habitat_sim.sensor import SensorType
import cv2 as cv
import numpy as np


from VisFly.utils.common import load_yaml_config
from envs.ObjectTrackingEnv import ObjectTrackingEnv

scene_num = 1
agent_num = 200

env = HoverEnv2(
    visual=False,
    num_scene=scene_num,
    num_agent_per_scene=agent_num,
    tensor_output=True,
)

scene_path = "VisFly/datasets/visfly-beta/configs/scenes/box15_wall_pillar"
sensor_kwargs = [{
    "sensor_type": SensorType.DEPTH,
    "uuid": "depth",
    "resolution": [64, 64],
    "position": [0, 0.2, 0.],
},
    # {
    #     "sensor_type": SensorType.SEMANTIC,
    #     "uuid": "semantic",
    #     "resolution": [64, 64],
    #     "position": [0, 0.2, 0.],
    # }
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
        "position": th.tensor([[0., 0, 5.5], [7, 0, 2.5]]),
        "line_width": 6.,
        "trajectory": True,
        "collision": True

    }
}

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

env = DynEnv(
    visual=False,
    num_scene=scene_num,
    num_agent_per_scene=agent_num,
    random_kwargs=random_kwargs,
    scene_kwargs=scene_kwargs,
    sensor_kwargs=sensor_kwargs,
    dynamics_kwargs={},
    tensor_output=True,
)


steps = 100
# env_config = load_yaml_config(f'exps/std/env_cfgs/objTracking.yaml')
# env = ObjectTrackingEnv(**env_config["eval_env"])
# env.reset()
env.reset()

start_time = time.time()
for i in range(steps):
    actions = th.rand((scene_num * agent_num, 4))
    env.step(actions)
    # img = env.render(is_draw_axes=True)
    # obs = env.sensor_obs["depth"]
    # semantic = env.sensor_obs["semantic"]
    # cv.imshow("img", cv.cvtColor(np.hstack(img), cv.COLOR_RGBA2BGRA))
    # cv.waitKey(100)

end_time = time.time()

elapsed_time = end_time - start_time
fps = (steps * scene_num * agent_num) / elapsed_time
print(f"Elapsed time for {steps} steps: {elapsed_time:.2f} seconds")
print(f"Approximate FPS: {fps:.2f}")
