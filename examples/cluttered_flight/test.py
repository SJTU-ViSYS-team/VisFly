import numpy as np

from VisFly.utils.evaluate import TestBase
import os, sys
from typing import Optional
from matplotlib import pyplot as plt
from VisFly.utils.FigFashion.FigFashion import FigFon


class Test(TestBase):
    def __init__(self,
                 model,
                 name,
                 save_path: Optional[str] = None,
                 ):
        # Ensure TestBase receives env and model correctly
        super(Test, self).__init__(env=model.env, model=model, name=name, save_path=save_path)

    def draw(self, names=None):
        state_data = [obs["state"] for obs in self.obs_all]
        state_data = np.array(state_data)
        t = np.array(self.t)[:, 0]
        for i in range(self.model.env.num_envs):
            fig = plt.figure(figsize=(5, 4))
            plt.subplot(2, 2, 1)
            plt.plot(t, state_data[:, i, 0:3], label=["x", "y", "z"])
            plt.legend()
            plt.subplot(2, 2, 2)
            plt.plot(t, state_data[:, i, 3:7], label=["w", "x", "y", "z"])
            plt.legend()
            plt.subplot(2, 2, 3)
            plt.plot(t, state_data[:, i, 7:10], label=["vx", "vy", "vz"])
            plt.legend()
            plt.subplot(2, 2, 4)
            plt.plot(t, state_data[:, i, 10:13], label=["wx", "wy", "wz"])
            plt.legend()
            plt.tight_layout()
            plt.show()
        col_dis = np.array([collision["col_dis"] for collision in self.collision_all])
        fig2, axes = FigFon.get_figure_axes(SubFigSize=(1, 1))
        axes.plot(t, col_dis)
        axes.set_xlabel("t/s")
        axes.set_ylabel("closest distance/m")
        plt.show()

        return [fig, fig2] 