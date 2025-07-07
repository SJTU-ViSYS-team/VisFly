
import sys
import os
import numpy as np
import torch
import time

sys.path.append(os.getcwd())
from utils.policies import extractors
from utils.algorithms.ppo import ppo
from utils import savers
import torch as th
from envs.NavigationEnv import NavigationEnv
from utils.launcher import rl_parser, training_params
from utils.type import Uniform

args = rl_parser().parse_args()
""" SAVED HYPERPARAMETERS """
training_params["num_env"] = 96
training_params["learning_step"] = 1e7
training_params["comment"] = args.comment
training_params["max_episode_steps"] = 256
training_params["n_steps"] = training_params["max_episode_steps"]
training_params["batch_size"] = training_params["num_env"] * training_params["n_steps"]
training_params["learning_rate"] = 1e-3
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/"

scene_path = "datasets/visfly-beta/configs/garage_simple_l_medium"

random_kwargs = {
    "state_generator_kwargs": [{
        "position": Uniform(mean=th.tensor([9., 0., 3.5]), half=th.tensor([8, 6., 3.])),
        "orientation": Uniform(mean=th.tensor([0., 0., 0.]), half=th.tensor([th.pi, th.pi, th.pi]))
    }]
}

num = 1024
env = NavigationEnv(num_agent_per_scene=num,
                    state_random_kwargs=random_kwargs,
                    visual=True,
                    max_episode_steps=training_params["max_episode_steps"],
                    scene_kwargs={
                        "path": scene_path,
                    },
                    )
img_i = 0
import cv2 as cv
path = os.getcwd() + "/datasets/Images/depth_dataset/64"
for i in range(100):
    obs = env.reset()
    depths = obs["depth"]
    # save gray image
    for depth in depths:
        cv.imwrite(f"{path}/depth_{img_i}.png", depth.squeeze())
        img_i +=1
    test = 1
