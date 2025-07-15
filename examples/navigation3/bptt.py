#!/usr/bin/env python3

import sys
import os
import time
import torch
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

# Parse args FIRST before importing any VisFly modules
from VisFly.utils.launcher import rl_parser, training_params
args = rl_parser().parse_args()

# Now import the rest after we know what mode we're in
from VisFly.utils.policies import extractors
from VisFly.utils.algorithms.BPTT import BPTT
from VisFly.utils import savers
from VisFly.envs.NavigationEnv import NavigationEnv3  # Use NavigationEnv3
from VisFly.utils.type import Uniform
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Disable gradient anomaly detection for physics-based BPTT
torch.autograd.set_detect_anomaly(True)  # Commented out for physics compatibility

""" SAVED HYPERPARAMETERS """
# Number of parallel environments (agents)
training_params["num_env"] = 150
# Total learning steps
training_params["learning_step"] = 1e7
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
                "position": {"mean": [0., 0., 1.], "half": [8., 8., 1.]},  # Increased from [1., 1., 1.] to [8., 8., 1.]
                "orientation": {"mean": [0., 0., 0.], "half": [0.1, 0.1, 0.1]},
                "velocity": {"mean": [0., 0., 0.], "half": [0.1, 0.1, 0.1]},
                "angular_velocity": {"mean": [0., 0., 0.], "half": [1., 1., 1.]},
            }
        ]
    }
}

# Scene configuration for visual rendering
scene_kwargs = {
    "path": "VisFly/datasets/spy_datasets/configs/garage_simple_l_medium"
}

# Dynamics configuration 
dynamics_kwargs = {
    "dt": 0.02,
    "ctrl_dt": 0.02,
    "action_type": "bodyrate",
    "ctrl_delay": True,
    # Use the calibrated drone parameters of Intel RealSense D435i quadrotor
    "cfg": "drone/drone_d435i",
}

def main():
    # Training mode
    if args.train:
        # Create NavigationEnv3 with gradients enabled for BPTT and visual=True
        env = NavigationEnv3(
            num_agent_per_scene=int(training_params["num_env"]),
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            visual=True,  # Enable visual for depth sensor
            requires_grad=True,
            max_episode_steps=int(training_params["max_episode_steps"]),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        env.reset()
        
        # Create separate evaluation environment (without gradients for efficiency)
        print("Creating evaluation environment...")
        eval_env = NavigationEnv3(
            num_agent_per_scene=int(training_params["num_env"]),
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            visual=True,  # Enable visual for depth sensor
            requires_grad=False,  # No gradients needed for evaluation
            max_episode_steps=int(training_params["max_episode_steps"]),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        eval_env.reset()
        
        # Load pretrained model if provided
        if args.weight is not None:
            model = BPTT.load(path=os.path.join(save_folder + args.weight))
        else:
            # Instantiate BPTT algorithm
            model = BPTT(
                env=env,
                policy="MultiInputPolicy",
                policy_kwargs=dict(
                    features_extractor_class=extractors.StateTargetImageExtractor,
                    features_extractor_kwargs=dict(
                        net_arch=dict(
                            depth=dict(layer=[128]),
                            state=dict(layer=[128, 64]),
                            target=dict(layer=[128, 64]),
                        ),
                        activation_fn=torch.nn.ReLU,
                    ),
                    net_arch=dict(pi=[64, 64], qf=[64, 64]),
                    activation_fn=torch.nn.ReLU,
                    optimizer_kwargs=dict(weight_decay=1e-5),
                ),
                learning_rate=training_params["learning_rate"],
                comment=args.comment,
                save_path=save_folder,
                horizon=int(training_params["horizon"]),
                gamma=training_params.get("gamma", 0.99),
                device="cuda" if torch.cuda.is_available() else "cpu",
                seed=int(training_params["seed"]),
            )
            
        # Manually set the eval_env since deepcopy doesn't work with complex environments
        model.eval_env = eval_env
        print("Evaluation environment successfully created and assigned.")
        
        # Train
        start_time = time.time()
        model.learn(int(training_params["learning_step"]))
        model.save()
        training_params["time"] = time.time() - start_time
        savers.save_as_csv(save_folder + "training_params_bptt.csv", training_params)
    else:
        # Testing mode
        test_model_path = save_folder + args.weight
        
        # Add render settings for test environment
        test_scene_kwargs = scene_kwargs.copy()
        
        # Option 1: Balanced angled view to capture drone flight space
        test_scene_kwargs["render_settings"] = {
            "mode": "fix",
            "view": "custom",
            "resolution": [1080, 1920],
            "position": torch.tensor([[14., 0., 6.], [0., -2., 0.]]),  # Angled position for good coverage
            "line_width": 8.,
            "trajectory": True,
            "axes": False,
        }
        
        # Option 2: Side view for wide horizontal coverage (alternative)
        # test_scene_kwargs["render_settings"] = {
        #     "mode": "fix",
        #     "view": "custom",
        #     "resolution": [1080, 1920],
        #     "position": torch.tensor([[0., 5., 10.], [0., 2., 0.]]),  # Back view for wide coverage
        #     "line_width": 8.,
        #     "trajectory": True,
        #     "axes": False,
        # }
        
        from tst import Test
        env = NavigationEnv3(
            num_agent_per_scene=1,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=test_scene_kwargs,  # Use test scene kwargs with render settings
            visual=True,  # Enable visual for depth sensor
            max_episode_steps=int(training_params["max_episode_steps"]),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        env.reset()
        # Initialize BPTT model for testing
        model = BPTT(
            env=env,
            policy="MultiInputPolicy",
            policy_kwargs=dict(
                features_extractor_class=extractors.StateTargetImageExtractor,
                features_extractor_kwargs=dict(
                    net_arch=dict(
                        depth=dict(layer=[128]),
                        state=dict(layer=[128, 64]),
                        target=dict(layer=[128, 64]),
                    ),
                    activation_fn=torch.nn.ReLU,
                ),
                net_arch=dict(pi=[64, 64], qf=[64, 64]),
                activation_fn=torch.nn.ReLU,
                optimizer_kwargs=dict(weight_decay=1e-5),
            ),
        )
        print(f"Loading model from: {test_model_path}")
        model.load(test_model_path)
        # Wrap model for test
        class ModelWrapper:
            def __init__(self, policy):
                self.policy = policy
                self.env = env
        wrapped = ModelWrapper(model.policy)
        test_handle = Test(
            env,  # First parameter: env
            wrapped,  # Second parameter: model (with .policy attribute)
            "bptt_test",  # Third parameter: name
            os.path.dirname(os.path.realpath(__file__)) + "/saved/test",  # Fourth parameter: save_path
        )
        # --- Extended evaluation -------------------------------------------------
        n_eval_episodes = 4  # how many complete episodes to evaluate for SR alignment
        aggregated_success = 0
        total_agents_eval = env.num_envs * n_eval_episodes

        for ep_i in range(n_eval_episodes):
            # create a sub-directory for this episode:  .../bptt_test/episode_XX/
            episode_dir = os.path.join(test_handle.save_path, f"episode_{ep_i:03d}")
            if not os.path.exists(episode_dir):
                os.makedirs(episode_dir, exist_ok=True)

            # Clear accumulated data from previous episodes
            test_handle.obs_all = []
            test_handle.state_all = []
            test_handle.info_all = []
            test_handle.action_all = []
            test_handle.collision_all = []
            test_handle.render_image_all = []
            test_handle.reward_all = []
            test_handle.t = []
            test_handle.eq_r = []
            test_handle.eq_l = []

            figs, _, _, _ = test_handle.test(
                is_fig=True,              # draw trajectory plots every episode
                is_fig_save=False,        # disable auto-save to avoid conflicts
                is_video=True,            # enable video rendering
                is_video_save=True,       # save videos
                is_sub_video=True,        # enable individual agent views (depth/color sensors)
            )

            # save all figures manually into the episode folder with meaningful names
            for fig_idx, fig in enumerate(figs):
                fig_path = os.path.join(episode_dir, f"trajectory_plot_{fig_idx}.png")
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                print(f"Episode {ep_i}: fig saved in {fig_path}")

            # Save videos (combined, global, and individual agent views)
            test_handle.save_combined_video(episode_dir)

            # Count agents that achieved success at any point during the episode
            successful_agents = set()
            
            for timestep_idx, timestep_info in enumerate(test_handle.info_all):
                for agent_idx, agent_info in enumerate(timestep_info):
                    if agent_info and "is_success" in agent_info and agent_info["is_success"]:
                        successful_agents.add(agent_idx)
            
            episode_successes = len(successful_agents)
            aggregated_success += episode_successes
            print(f"Episode {ep_i}: {episode_successes}/{env.num_envs} agents reached target at some point")

        eval_sr = aggregated_success / total_agents_eval
        print(f"\nAggregated evaluation over {n_eval_episodes} episodes → Success Rate: {eval_sr:.3f}\n")

if __name__ == "__main__":
    main() 