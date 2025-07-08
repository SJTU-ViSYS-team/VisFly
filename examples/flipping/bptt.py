#!/usr/bin/env python3

import sys
import os
import time
import torch
import numpy as np
from stable_baselines3.common.torch_layers import CombinedExtractor

# Ensure project root is in path
sys.path.append(os.getcwd())

from VisFly.utils.algorithms.BPTT import BPTT
from VisFly.utils import savers
from VisFly.envs.FlippingEnv import FlippingEnv
from VisFly.utils.launcher import rl_parser, training_params
from VisFly.utils.type import Uniform

args = rl_parser().parse_args()

""" SAVED HYPERPARAMETERS """
# Number of parallel environments (agents)
training_params["num_env"] = 48
# Total learning steps
training_params["learning_step"] = 1e6
# Comments and seed
training_params["comment"] = args.comment
training_params["seed"] = args.seed
# Episode length
training_params["max_episode_steps"] = 256
# Learning rate for BPTT
training_params["learning_rate"] = 1e-3
# BPTT horizon (how many steps to backprop through)
training_params["horizon"] = training_params["max_episode_steps"]

# Directory where to save checkpoints and logs
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/"

# Random initialization for environment resets
random_kwargs = {
    "state_generator": {
        "class": "Uniform",
        "kwargs": [
            {
                "position": {"mean": [0., 0., 1.], "half": [0., 0., 0.]},
                # Orientation specified as Euler angles (roll, pitch, yaw)
                "orientation": {"mean": [0., 0., 0.], "half": [0.1, 0.1, 0.1]},
                "velocity": {"mean": [0., 0., 0.], "half": [0., 0., 0.]},
                "angular_velocity": {"mean": [0., 0., 0.], "half": [1., 1., 1.]}
            }
        ]
    }
}

dynamics_kwargs = {
    "ori_output_type": "euler"
}
def main():
    if args.train:
        # Create flipping environment
        env = FlippingEnv(
            num_agent_per_scene=int(training_params["num_env"]),
            random_kwargs=random_kwargs,
            visual=False,
            max_episode_steps=int(training_params["max_episode_steps"]),
            requires_grad=True,  # Enable gradients for BPTT algorithm
            # dynamics_kwargs=dynamics_kwargs,
        )
        # Initialize environment before training
        env.reset()
        # If a pretrained weight is specified, load it
        if args.weight is not None:
            model = BPTT.load(path=os.path.join(save_folder + args.weight))
        else:
            # Instantiate BPTT algorithm
            model = BPTT(
                env=env,
                policy="MultiInputPolicy",
                policy_kwargs=dict(
                    net_arch=[64, 64],
                    activation_fn=torch.nn.ReLU,
                    optimizer_kwargs=dict(weight_decay=1e-5),
                    features_extractor_class=CombinedExtractor,
                ),
                learning_rate=training_params["learning_rate"],
                logger_kwargs={},  # custom logger settings if needed
                comment=args.comment,
                save_path=save_folder,
                horizon=training_params["horizon"],
                gamma=training_params.get("gamma", 0.99),
                device="cuda" if torch.cuda.is_available() else "cpu",
                seed=int(training_params["seed"]),
            )
        # Train
        start_time = time.time()
        model.learn(int(training_params["learning_step"]))
        model.save()
        # Record total time
        training_params["time"] = time.time() - start_time
        savers.save_as_csv(save_folder + "training_params_bptt.csv", training_params)
    else:
        # Test mode
        test_model_path = save_folder + args.weight
        from test import Test
        env = FlippingEnv(
            num_agent_per_scene=1,
            random_kwargs=random_kwargs,
            visual=False,  # disable visual to avoid render errors
            max_episode_steps=int(training_params["max_episode_steps"]),
            # dynamics_kwargs=dynamics_kwargs,
        )
        # Initialize environment before testing
        env.reset()
        # Load and test the trained model
        model = BPTT(env=env, policy="MultiInputPolicy", policy_kwargs={})
        print(f"Loading model from: {test_model_path}")
        model.load(test_model_path)
        print(f"Model loaded. Policy type: {type(model.policy)}")
        print(f"Policy attributes: {dir(model.policy)}")
        
        # Create a wrapper object that has a policy attribute for the test
        class ModelWrapper:
            def __init__(self, policy):
                self.policy = policy
                self.env = env
        
        wrapped_model = ModelWrapper(model.policy)
        print(f"Wrapped model policy type: {type(wrapped_model.policy)}")
        test_handle = Test(
            wrapped_model,
            "bptt_test",
            os.path.dirname(os.path.realpath(__file__)) + "/saved/test",
        )
        print(f"Test handle model type: {type(test_handle.model)}")
        print(f"Test handle model: {test_handle.model}")
        if hasattr(test_handle.model, 'policy'):
            print(f"Test handle model policy type: {type(test_handle.model.policy)}")
        else:
            print("Test handle model has no policy attribute")
        test_handle.test(
            is_fig=True,
            is_fig_save=True,
            is_video=False,  # Disable video for now
            is_video_save=False,  # Disable video save for now
        )

if __name__ == "__main__":
    main() 