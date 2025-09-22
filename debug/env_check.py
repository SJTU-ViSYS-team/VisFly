import numpy as np
import torch as th
import cv2

"""
    import test env
"""
from exps.std.run import main


env = main(debug_env=True)
env.reset()

FPS_num = 1000

num_agent = env.num_agent
for i in range(FPS_num):
    action = th.randn((num_agent, 4)).clamp(-1,1)
    env.step(action)
    img = env.render()

    cv2.imshow("render", cv2.cvtColor(img[0], cv2.COLOR_RGBA2BGRA))
    cv2.waitKey(1)

