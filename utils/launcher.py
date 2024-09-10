import argparse

training_params = {
    "time": None,
    "num_envs": 96,
    "learning_step": 2e7,
    "policy": None,
    "algorithm": "PPO",
    "learning_rate": 3e-4,
    "n_steps": 256,
    "batch_size": 96*256,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.00,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "clip_range": 0.2,
    "n_epochs": 10,
    "target_kl": None,
    "verbose": 1,
    "comment": None,
    "seed": 42,
    "max_episode_steps": 256,
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
    parser.add_argument("-m", "--num", type=int, default=100, help="number of agents")
    parser.add_argument("-hl", "--horizon", type=int, default=40, help="horizon")
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
    parser.add_argument("-m", "--num", type=int, default=100, help="number of agents")
    return parser

