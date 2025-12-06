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

# print mesh
if True:
    mesh = env.envs.sceneManager.scenes[0].scene_mesh

faces_id = th.tensor(mesh.faces).reshape(-1, 3)
vertices = th.tensor(mesh.vertices)


