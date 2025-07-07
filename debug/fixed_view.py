import os
import sys
sys.path.append('..')

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

# Try to use a simpler scene or disable visual if scene loading fails
try:
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
            "position": th.tensor([[7., 6.8, 5.5], [7, 4.8, 4.5]]),
            "line_width": 6.,
            "trajectory": True,
        }
    }
    visual = True
except Exception as e:
    print(f"Scene loading failed, using non-visual mode: {e}")
    sensor_kwargs = []
    scene_kwargs = {}
    visual = False

# Add proper dynamics_kwargs
dynamics_kwargs = {
    "action_type": "bodyrate",
    "ori_output_type": "quaternion",
    "dt": 0.005,
    "ctrl_dt": 0.03,
    "ctrl_delay": True,
    "comm_delay": 0.09,
    "action_space": (-1, 1),
    "integrator": "euler",
    "drag_random": 0,
}

num_agent = 4
env = NavigationEnv2(
    visual=visual,
    num_scene=1,
    num_agent_per_scene=num_agent,
    random_kwargs=random_kwargs,
    dynamics_kwargs=dynamics_kwargs,
    scene_kwargs=scene_kwargs,
    sensor_kwargs=sensor_kwargs,
)

print("Environment created successfully!")
env.reset()
print("Environment reset successfully!")

t = 0
max_steps = 100  # Limit the number of steps for testing
while t < max_steps:
    try:
        a = th.rand((num_agent, 4))
        env.step(a)
        
        if visual:
            img = env.render(is_draw_axes=True)
            obs = env.sensor_obs["depth"]
            cv.imshow("img", img[0])
            cv.imshow("obs", obs[0][0])
            cv.waitKey(100)
        else:
            print(f"Step {t}: Position = {env.position[0]}")
        
        t += 1
    except KeyboardInterrupt:
        print("Interrupted by user")
        break
    except Exception as e:
        print(f"Error at step {t}: {e}")
        break

if visual:
    cv.destroyAllWindows()
print("Test completed!") 