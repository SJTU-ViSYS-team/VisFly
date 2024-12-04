#!/usr/bin/env python3

import sys
import os
sys.path.append(os.getcwd())

import torch
import time
import yaml

from VisFly.utils.policies import extractors
import torch as th
from VisFly.envs.HoverEnv import HoverEnv2
from VisFly.utils.launcher import rl_parser, training_params
from VisFly.utils.algorithms.BPTT import BPTT

args = rl_parser().parse_args()

save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/"
scene_path = "VisFly/datasets/spy_datasets/configs/garage_empty"


def main():
    with open(os.path.dirname(os.path.abspath(__file__))+'/cfg/bptt.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # if train mode, train the model
    if args.train:

        env = HoverEnv2(
            **config["env"]
        )

        model = BPTT(
            env=env,
            seed=args.seed,
            comment=args.comment,
            **config["algorithm"]
        )

        if args.weight is not None:
            model.load(path=save_folder + args.weight)

        model.learn(**config["learn"])
        model.save()

    # Testing mode with a trained weight
    else:
        test_model_path = save_folder + args.weight
        from test import Test
        env = HoverEnv2(**config["eval_env"])
        model = BPTT.load(test_model_path, env=env)

        test_handle = Test(
            model=model,
            save_path=os.path.dirname(os.path.realpath(__file__)) + "/saved/test",
            name=args.weight)
        test_handle.test(**config["test"])


if __name__ == "__main__":
    main()
