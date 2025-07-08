#!/usr/bin/env python3

import sys
import os
import time
import torch
import numpy as np


# Ensure project root is in path
sys.path.append(os.getcwd())
from VisFly.utils.policies import extractors
from VisFly.utils.algorithms.ppo import ppo
from VisFly.utils import savers
from VisFly.envs.NavigationEnv import NavigationEnv3
from VisFly.utils.launcher import rl_parser, training_params
from VisFly.utils.type import Uniform
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args = rl_parser().parse_args()

""" SAVED HYPERPARAMETERS """
# Number of parallel environments
training_params["num_env"] = 96
# Total learning steps
training_params["learning_step"] = 1e7
# Comments and seed
training_params["comment"] = args.comment
training_params["seed"] = args.seed
# Episode settings
training_params["max_episode_steps"] = 256
training_params["n_steps"] = training_params["max_episode_steps"]
training_params["batch_size"] = training_params["num_env"] * training_params["n_steps"]
# Learning rate
training_params["learning_rate"] = 1e-3
training_params["n_epochs"] = 10
# Save folder
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/"

# Random initialization for environment resets
random_kwargs = {
    "state_generator": {
        "class": "Uniform",
        "kwargs": [
            {
                "position": {"mean": [0., 0., 1.], "half": [1., 1., 1.]},
                "orientation": {"mean": [0., 0., 0.], "half": [0.1, 0.1, 0.1]},
                "velocity": {"mean": [0., 0., 0.], "half": [0.1, 0.1, 0.1]},
                "angular_velocity": {"mean": [0., 0., 0.], "half": [1., 1., 1.]}
            }
        ]
    }
}

scene_path = "VisFly/datasets/spy_datasets/configs/garage_simple_l_medium"
scene_kwargs = {
    "path": scene_path,
}


def main():
    # Training mode
    if args.train:
        env = NavigationEnv3(
            num_agent_per_scene=int(training_params["num_env"]),
            random_kwargs=random_kwargs,
            visual=True,
            requires_grad=False,
            max_episode_steps=int(training_params["max_episode_steps"]),
            scene_kwargs=scene_kwargs
        )
        if args.weight:
            model = ppo.load(save_folder + args.weight, env=env)
        else:
            model = ppo(
                policy="MultiInputPolicy",
                policy_kwargs=dict(
                    features_extractor_class=extractors.StateTargetImageExtractor,
                    features_extractor_kwargs=dict(
                        net_arch=dict(
                            depth=dict(
                                layer=[128]
                            ),
                            state=dict(
                                layer=[128, 64]
                            ),
                            target=dict(
                                layer=[128, 64]
                            ),
                        ),
                        activation_fn=torch.nn.ReLU,
                    ),
                    net_arch=dict(
                        pi=[64, 64],
                        vf=[64, 64],
                    ),
                    activation_fn=torch.nn.ReLU,
                    optimizer_kwargs=dict(weight_decay=1e-5),
                ),
                env=env,
                verbose=training_params.get("verbose", 1),
                tensorboard_log=save_folder,
                gamma=training_params.get("gamma", 0.99),
                n_steps=training_params["n_steps"],
                ent_coef=training_params.get("ent_coef", 0.0),
                learning_rate=training_params["learning_rate"],
                vf_coef=training_params.get("vf_coef", 0.5),
                max_grad_norm=training_params.get("max_grad_norm", 0.5),
                batch_size=training_params["batch_size"],
                gae_lambda=training_params.get("gae_lambda", 0.95),
                n_epochs=training_params.get("n_epochs", 10),
                clip_range=training_params.get("clip_range", 0.2),
                device="cuda" if torch.cuda.is_available() else "cpu",
                seed=int(training_params["seed"]),
                comment=args.comment,
            )
        start_time = time.time()
        model.learn(int(training_params["learning_step"]))
        model.save()
        training_params["time"] = time.time() - start_time
        savers.save_as_csv(save_folder + "training_params.csv", training_params)
    else:
        # Testing/visualization mode
        test_model_path = save_folder + args.weight
        from tst import Test
        env = NavigationEnv3(
            num_agent_per_scene=1,
            random_kwargs=random_kwargs,
            visual=True,
            requires_grad=False,
            max_episode_steps=int(training_params["max_episode_steps"]),
            scene_kwargs=scene_kwargs
        )
        model = ppo.load(test_model_path, env=env)
        test_handle = Test(
            model=model,
            save_path=os.path.dirname(os.path.realpath(__file__)) + "/saved/test",
            name=args.weight
        )
        test_handle.test(
            is_fig=True,
            is_fig_save=True,
            is_render=True,
            is_video=True,
            is_video_save=True,
        )

if __name__ == "__main__":
    main() 