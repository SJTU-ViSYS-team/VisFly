import os, sys
sys.path.append(os.getcwd())
# from examples.stable_flight.droneStableEnvs import droneStableEnvs
from envs.NavigationEnv import NavigationEnv
from utils.launcher import dl_parser as parser
from utils.algorithms.dl_algorithm import ApgBase
from test import Test
from utils.policies import extractors
from utils.launcher import training_params
import torch

args = parser().parse_args()
training_params["horizon"] = 96
training_params["max_episode_steps"] = 256
training_params["num_env"] = 96

save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/"
path = "datasets/spy_datasets/configs/garage_simple"

# torch.autograd.set_detect_anomaly(True)


def main():

    if args.train:
        env = NavigationEnv(num_agent_per_scene=training_params["num_env"],
                            visual=True,
                            max_episode_steps=training_params["max_episode_steps"],
                            requires_grad=True,
                            scene_kwargs={
                                    "path": path,
                                },
                            device= "cuda"
                            )
        if args.weight is not None:
            model = ApgBase.load(env, save_folder + args.weight)
        else:
            model = ApgBase(
                env=env,
                policy="ActorPolicy",
                policy_kwargs=dict(
                    features_extractor_class=policies.StateTargetImageExtractor,
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
                            }
                        }
                    },
                    net_arch=dict(
                        pi=[64, 64],
                        vf=[]),
                    activation_fn=torch.nn.ReLU,
                    optimizer_kwargs=dict(weight_decay=1e-5)
                ),
                learning_rate=1e-3,
                device="cuda",
                commit=args.comment,
            )
        model.learn(total_timesteps=1e7,
                  horizon=training_params["horizon"])
        model.save()
    else:
        env = NavigationEnv(num_agent_per_scene=1, visual=True,
                            scene_kwargs={
                                 "path": path,
                                 "render_settings": {
                                     "mode": "fix",
                                     "view": "near",
                                     "resolution": [1080, 1080],
                                     # "position": torch.tensor([[8., 7., 3.]]),
                                 }
                             })
        model = ApgBase.load(env, save_folder + args.weight)
        test_handle = Test(
            env=env,
            policy=model,
            name="apg",
        )
        test_handle.test(is_fig=True, is_fig_save=True, is_render=True, is_video=True, is_video_save=True)

if __name__ == "__main__":
    main()

