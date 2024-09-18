#!/usr/bin/env python3

import sys
import os
import numpy as np
import torch
import time

sys.path.append(os.getcwd())
from VisFly.utils.policies import extractors
from VisFly.utils.algorithms.ppo import ppo
from VisFly.utils import savers
import torch as th
from VisFly.envs.MultiNavigationEnv import MultiNavigationEnv
from VisFly.utils.launcher import rl_parser, training_params
from VisFly.utils.type import Uniform

args = rl_parser().parse_args()
""" SAVED HYPERPARAMETERS """
training_params["num_env"] = 3
training_params["learning_step"] = 1e7
training_params["comment"] = args.comment
training_params["max_episode_steps"] = 256
training_params["n_steps"] = training_params["max_episode_steps"]
training_params["batch_size"] = training_params["num_env"] * training_params["n_steps"]
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/"
training_params["learning_rate"] = 1e-4

scene_path = "datasets/spy_datasets/configs/garage_crossing"


random_kwargs = {
    "state_generator":
        {
            "class": "Uniform",
            "kwargs":[
                {"position": {"mean": [1., 0., 1.5], "half": [0.0, 2., 1.]}},
            ]
        }
}



def main():
    # if train mode, train the model
    if args.train:
        env = MultiNavigationEnv(num_agent_per_scene=training_params["num_env"],
                                 num_scene=24,
                                 random_kwargs=random_kwargs,
                                 visual=True,
                                 max_episode_steps=training_params["max_episode_steps"],
                                 scene_kwargs={
                                     "path": scene_path,
                                 },
                                 )

        if args.weight is not None:
            model = ppo.load(save_folder + args.weight, env=env)
        else:
            model = ppo(
                policy="MultiInputPolicy",
                policy_kwargs=dict(
                    features_extractor_class=extractors.SwarmStateTargetImageExtractor,
                    features_extractor_kwargs={
                        "net_arch": {
                            "depth": {

                                "mlp_layer": [128],
                            },
                            "state": {
                                "mlp_layer": [128, 64],
                            },
                            "target": {
                                "mlp_layer": [128, 64],
                            },
                            "swarm": {
                                "mlp_layer": [128, 64],
                            }
                        }
                    },
                    net_arch=dict(
                        pi=[64, 64],
                        vf=[64, 64]),
                    activation_fn=torch.nn.ReLU,
                    optimizer_class=torch.optim.Adam,
                    optimizer_kwargs=dict(weight_decay=1e-5)
                ),
                env=env,
                verbose=training_params["verbose"],
                tensorboard_log=save_folder,
                gamma=training_params["gamma"],  # lower 0.9 ~ 0.99
                n_steps=training_params["n_steps"],
                ent_coef=training_params["ent_coef"],
                learning_rate=training_params["learning_rate"],
                vf_coef=training_params["vf_coef"],
                max_grad_norm=training_params["max_grad_norm"],
                batch_size=training_params["batch_size"],
                gae_lambda=training_params["gae_lambda"],
                n_epochs=training_params["n_epochs"],
                clip_range=training_params["clip_range"],
                device="cuda",
                seed=training_params["seed"],
                comment=args.comment,
            )

        start_time = time.time()
        model.learn(training_params["learning_step"])
        model.save()
        training_params["time"] = time.time() - start_time

        savers.save_as_csv(save_folder + "training_params.csv", training_params)

    # Testing mode with a trained weight
    else:
        test_model_path = save_folder + args.weight
        from test import Test
        env = MultiNavigationEnv(num_agent_per_scene=3, visual=True,
                                 random_kwargs=random_kwargs,
                                 scene_kwargs={
                                     "path": scene_path,
                                     "render_settings": {
                                         "mode": "fix",
                                         "view": "top",
                                         "resolution": [1080, 1920],
                                         # "position": th.tensor([[6., 6.8, 5.5], [6,4.8,4.5]]),
                                         # "position": th.tensor([[7., 6.8, 5.5], [7, 4.8, 4.5]]),
                                         "position": th.tensor([[7., 0, 7.5]]),
                                         "trajectory": True,
                                         "line_width": 6
                                     }
                                 })
        model = ppo.load(test_model_path, env=env)

        test_handle = Test(
            model=model,
            save_path=os.path.dirname(os.path.realpath(__file__)) + "/saved/test",
            name=args.weight)
        test_handle.test(is_fig=True, is_fig_save=True, is_render=True, is_video=True, is_video_save=True,
                         render_kwargs={
                             # "points": th.tensor([[13., 0, 1],
                             #                      [1, 0, 1],
                             #                      ])
                         })


if __name__ == "__main__":
    main()
