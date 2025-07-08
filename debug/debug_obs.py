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

random_kwargs = {
    "state_generator":
        {
            "class": "Uniform",
            "kwargs": [
                {"position": {"mean": [1., 0., 1.5], "half": [2.0, 2.0, 1.0]}},
            ]
        }
}

# Use a working scene path
scene_path = "../datasets/spy_datasets/configs/garage_simple_l_medium"
sensor_kwargs = [{
    "sensor_type": SensorType.DEPTH,  # Changed to DEPTH for consistency
            "uuid": "depth",
    "resolution": [64, 64],  # Reduced resolution for testing
            "position": [0,0.2,0.],
        }]
scene_kwargs = {
    "path": scene_path,
    "render_settings": {
        "mode": "fix",
        "view": "custom",
        "resolution": [1080, 1920],
        "position": th.tensor([[7., 6.8, 5.5], [7, 4.8, 4.5]]),
        "line_width": 6.,
        "trajectory": True,
    }
}

try:
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
    print("Environment created and reset successfully!")

t = 0
    max_steps = 50  # Limit steps for testing
    while t < max_steps:
    a = th.rand((num_agent, 4))
    env.step(a)
        
        try:
    img = env.render(is_draw_axes=True)
            print(f"Step {t}: Position = {env.position[0]}")
            
            if "depth" in env.sensor_obs:
    obs = env.sensor_obs["depth"]
    cv.imshow("img", img[0])
    cv.imshow("obs", obs[0][0])
    cv.waitKey(100)
        except Exception as e:
            print(f"Render error at step {t}: {e}")
            break
            
        t += 1

    cv.destroyAllWindows()
    print("Test completed successfully!")

except Exception as e:
    print(f"Error creating environment: {e}")
    import traceback
    traceback.print_exc()



