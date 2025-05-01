import argparse

training_params = {
    "time": None,
    "num_envs": 100,
    "learning_step": 0.2e7,
    "policy": None,
    "algorithm": "PPO",
    "learning_rate": 3e-4,
    "n_steps": 100,
    "batch_size": 10000,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.00,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "clip_range": 0.2,
    "n_epochs": 5,
    "target_kl": None,
    "verbose": 1,
    "comment": None,
    "seed": 42,
    "max_episode_steps": 256,
    "train_freq": 100,
}

def dl_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", "-t",
        type=int,
        default=1,
        help="To train new model or simply test pre-trained model",
    )
    parser.add_argument("--render", type=int, default=0, help="Not Implemented Yet")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./saved/",
        help="Directory where to save the checkpoints and training metrics",
    )
    parser.add_argument(
        "-w", "--weight", type=str, default=None, help="trained weight name"
    )
    parser.add_argument("-c", "--comment", type=str, default=None, help="add comments")
    parser.add_argument("--log", type=str, default="")
    return parser


def rl_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train","-t",
        type=int,
        default=1,
        help="To train new model or simply test pre-trained model",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./saved/",
        help="Directory where to save the checkpoints and training metrics",
    )
    parser.add_argument(
        "-w", "--weight", type=str, default=None, help="trained weight name"
    )
    parser.add_argument("-c", "--comment", type=str, default=None, help="add comments")
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("-s", "--seed", type=int, default=42, help="seed")
    parser.add_argument("-a", "--algorithm", type=str, default="ppo", help="select the RL algorithm")
    parser.add_argument(
        "--env","-e",
        type=str,
        default="navigation",
        help="To train new model or simply test pre-trained model",
    )
    #debug add such --logdir /saved/original_with_new_trainAC --task dmc_walker_walk_3 --configs dmc_vision
    parser.add_argument("--logdir", type=str, default="/saved/original_with_new_trainAC")
    parser.add_argument("--task", type=str, default="dmc_walker_walk_3")
    parser.add_argument("--configs",type=str, default="dmc_control")
    return parser

