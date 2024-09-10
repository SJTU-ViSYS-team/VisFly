import numpy as np

from utils.evaluate import TestBase
import os, sys
from typing import Optional
from matplotlib import pyplot as plt
from utils.FigFashion.FigFashion import FigFon
class Test(TestBase):
    def __init__(self,
                model,
                 name,
        save_path: Optional[str] = None,
                 ):

        super(Test, self).__init__(model, name, save_path, )

    def draw(self, names=None):
        state_data = [obs["state"] for obs in self.obs_all]
        state_data = np.array(state_data)
        t = np.array(self.t)
        # fig = plt.figure(figsize=(5,4))
        # plt.subplot(2, 2, 1)
        # plt.plot(t, state_data[:,0, 0:3], label=["x", "y", "z"])
        # plt.legend()
        # plt.subplot(2, 2, 2)
        # plt.plot(t, state_data[:,0, 3:7], label=["w", "x", "y", "z"])
        # plt.legend()
        # plt.subplot(2, 2, 3)
        # plt.plot(t, state_data[:,0, 7:10], label=["vx", "vy", "vz"])
        # plt.legend()
        # plt.subplot(2, 2, 4)
        # plt.plot(t, state_data[:,0, 10:13], label=["wx", "wy", "wz"])
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        # col_dis = np.array([collision["col_dis"] for collision in self.collision_all])
        # fig2,axes = FigFon.get_figure_axes(SubFigSize=(1,1))
        # axes.plot(t, col_dis)
        # axes.set_xlabel("t/s")
        # axes.set_ylabel("closest distance/m")
        # plt.show()
        pass
        return []
        # return  [fig, fig2]


