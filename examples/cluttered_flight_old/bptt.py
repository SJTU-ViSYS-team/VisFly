#!/usr/bin/env python3

"""BPTT training entry-point for the cluttered_flight_old scenario.

This mirrors `examples/navigation3/bptt.py`, but re-uses the random
initialisation and scene used by the PPO example in the same folder.
The script trains with the differentiable dynamics (requires_grad=True)
so that gradients can flow through the environment.
"""

import os
import sys
import time
import torch
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

from VisFly.utils.launcher import rl_parser, training_params

# Parse CLI first (sets train/test mode, seed, etc.)
args = rl_parser().parse_args()

# Enable autograd anomaly detection when training for easier debugging of in-place ops
import torch
if args.train:
    torch.autograd.set_detect_anomaly(True)

from VisFly.utils.policies import extractors
from VisFly.utils.algorithms.BPTT import BPTT
from VisFly.utils import savers
from VisFly.envs.NavigationEnv import NavigationEnv2  # Use analytical env variant
from habitat_sim.sensor import SensorType

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------- Hyper-parameters -----------------------------------
training_params.update({
    "num_env": 96,                          # match PPO example
    "learning_step": 1e7,
    "comment": args.comment,
    "seed": args.seed,
    "max_episode_steps": 256,
    "learning_rate": 1e-3,
    "horizon": 256,                         # BPTT horizon – back-prop through full episode
})

save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")

# ----------------------- Environment set-up ---------------------------------
scene_path = "VisFly/datasets/spy_datasets/configs/garage_simple_l_medium"

random_kwargs = {
    "state_generator": {
        "class": "Uniform",
        "kwargs": [
            {"position": {"mean": [1.0, 0.0, 1.5], "half": [0.0, 2.0, 1.0]}},
        ],
    },
}

# Depth sensor for visual input
sensor_kwargs = [
    {
        "sensor_type": SensorType.DEPTH,
        "uuid": "depth",
        "resolution": [64, 64],
    }
]

dynamics_kwargs = {
    "dt": 0.02,
    "ctrl_dt": 0.02,
    "action_type": "bodyrate",
    "ctrl_delay": True,
    "cfg": "drone/drone_d435i",
}

scene_kwargs = {"path": scene_path}


def main():
    if args.train:
        # ---------------- Training environment (gradients enabled) ----------
        env = NavigationEnv2(
            num_agent_per_scene=int(training_params["num_env"]),
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
            visual=True,
            requires_grad=True,
            max_episode_steps=int(training_params["max_episode_steps"]),
            device=device,
        )
        env.reset()
        env.to(device)

        # ---------------- Evaluation environment (no gradients) -------------
        eval_env = NavigationEnv2(
            num_agent_per_scene=int(training_params["num_env"]),
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
            visual=True,
            requires_grad=False,
            max_episode_steps=int(training_params["max_episode_steps"]),
            device=device,
        )
        eval_env.reset()
        eval_env.to(device)

        # ---------------- Load / initialise BPTT model -----------------------
        if args.weight is not None:
            model = BPTT.load(path=os.path.join(save_folder, args.weight))
        else:
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
                horizon=int(training_params["horizon"]),
                gamma=training_params.get("gamma", 0.99),
                comment=args.comment,
                save_path=save_folder,
                device=device,
                seed=int(training_params["seed"]),
            )
        # Attach separate evaluation env (deepcopy doesn’t work for complex envs)
        model.eval_env = eval_env

        # ---------------- Train ------------------------------------------------
        start = time.time()
        model.learn(int(training_params["learning_step"]))
        model.save()
        training_params["time"] = time.time() - start
        savers.save_as_csv(os.path.join(save_folder, "training_params_bptt.csv"), training_params)

    else:
        # ---------------- Test mode -----------------------------------------
        test_model_path = os.path.join(save_folder, args.weight)

        from tst import Test  # use the enhanced tester we patched

        env = NavigationEnv2(
            num_agent_per_scene=1,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=dict(
                path=scene_path,
                render_settings={
                    "mode": "fix",
                    "view": "custom",
                    "resolution": [1080, 1920],
                    "position": torch.tensor([[6.0, 6.8, 5.5], [6.0, 4.8, 4.5]]),
                    "line_width": 6.0,
                    "trajectory": True,
                },
            ),
            sensor_kwargs=sensor_kwargs,
            visual=True,
            requires_grad=False,
            max_episode_steps=int(training_params["max_episode_steps"]),
            device=device,
        )
        env.reset()
        env.to(device)

        # Minimal BPTT wrapper just to load the policy
        model = BPTT(env=env)
        print(f"Loading BPTT model from: {test_model_path}")
        model.load(test_model_path)

        class Wrapper:
            def __init__(self, policy):
                self.policy = policy
                self.env = env

        wrapped = Wrapper(model.policy)
        tester = Test(env, wrapped, name=args.weight, save_path=os.path.join(save_folder, "test"))
        tester.test(is_fig=True, is_fig_save=True, is_video=True, is_video_save=True)


if __name__ == "__main__":
    main() 