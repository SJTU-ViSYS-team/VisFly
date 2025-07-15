#!/usr/bin/env python3

import sys
import os
import time
import torch
import numpy as np

sys.path.append(os.getcwd())
from VisFly.utils.algorithms.ppo import ppo
from VisFly.utils import savers
from VisFly.envs.FlippingEnv import FlippingEnv
from VisFly.utils.launcher import rl_parser, training_params
from VisFly.utils.type import Uniform
from stable_baselines3.common.torch_layers import CombinedExtractor

args = rl_parser().parse_args()

""" SAVED HYPERPARAMETERS """
training_params["num_env"] = 48
training_params["learning_step"] = 1e7
training_params["comment"] = args.comment
training_params["max_episode_steps"] = 256
training_params["n_steps"] = training_params["max_episode_steps"]
training_params["batch_size"] = training_params["num_env"] * training_params["n_steps"]
training_params["learning_rate"] = 1e-4

save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/"

random_kwargs = {
    "state_generator": {
        "class": "Uniform",
        "kwargs": [
            {
                "position": {"mean": [0., 0., 1.], "half": [0., 0., 0.]},
                "orientation": {"mean": [0., 0., 0.], "half": [0.1, 0.1, 0.1]},
                "velocity": {"mean": [0., 0., 0.], "half": [0., 0., 0.]},
                "angular_velocity": {"mean": [0., 0., 0.], "half": [1., 1., 1.]},
            }
        ]
    }
}


def main():
    if args.train:
        env = FlippingEnv(
            num_agent_per_scene=training_params["num_env"],
            random_kwargs=random_kwargs,
            visual=False,
            max_episode_steps=training_params["max_episode_steps"],
        )
        env.reset()
        if args.weight is not None:
            model = ppo.load(save_folder + args.weight, env=env)
        else:
            model = ppo(
                policy="MultiInputPolicy",
                policy_kwargs=dict(
                    net_arch=[64, 64],
                    activation_fn=torch.nn.ReLU,
                    features_extractor_class=CombinedExtractor,
                    optimizer_kwargs=dict(weight_decay=1e-5),
                ),
                env=env,
                verbose=training_params.get("verbose", 1),
                tensorboard_log=save_folder,
                gamma=training_params.get("gamma", 0.99),
                n_steps=training_params.get("n_steps", training_params["max_episode_steps"]),
                ent_coef=training_params.get("ent_coef", 0.0),
                learning_rate=training_params.get("learning_rate", 1e-4),
                vf_coef=training_params.get("vf_coef", 0.5),
                max_grad_norm=training_params.get("max_grad_norm", 0.5),
                batch_size=training_params.get("batch_size", training_params["num_env"] * training_params["max_episode_steps"]),
                gae_lambda=training_params.get("gae_lambda", 0.95),
                n_epochs=training_params.get("n_epochs", 5),
                clip_range=training_params.get("clip_range", 0.2),
                device="cuda" if torch.cuda.is_available() else "cpu",
                seed=int(training_params.get("seed", 42)),
                comment=args.comment,
            )
        start_time = time.time()
        model.learn(training_params["learning_step"])
        model.save()
        training_params["time"] = time.time() - start_time
        savers.save_as_csv(save_folder + "training_params.csv", training_params)
    else:
        test_model_path = save_folder + args.weight
        from test import Test
        env = FlippingEnv(
            num_agent_per_scene=1,
            random_kwargs=random_kwargs,
            visual=False,
            max_episode_steps=training_params["max_episode_steps"],
        )
        env.reset()
        model = ppo.load(test_model_path, env=env)
        test_handle = Test(
            model,
            args.weight,
            os.path.dirname(os.path.realpath(__file__)) + "/saved/test",
        )
        test_handle.test(
            is_fig=True,
            is_fig_save=True,
            is_video=False,
            is_video_save=False,
        )


if __name__ == "__main__":
    main() 