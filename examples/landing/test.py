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
        super(Test, self).__init__(model, name, save_path, )

    def draw(self, names=None):
        state_data = [obs["state"] for obs in self.obs_all]
        state_data = np.array(state_data)
        self.reward_all = np.array(self.reward_all).T
        t = np.array(self.t)[:,0]
        for i in range(self.model.env.num_envs):
            fig = plt.figure(figsize=(5, 4))
            ax1 = plt.subplot(2, 2, 1)
            plt.plot(t, state_data[:, i, 0:3], label=["x", "y", "z"])
            plt.legend()
            ax2 = ax1.twinx()
            ax2.plot(t[1:], self.reward_all[i], 'r-')  # 'r-' 代表红色实线
            ax2.set_ylabel('Logarithmic', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
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
            plt.suptitle(str(i))
            plt.show()

        col_dis = np.array([collision["col_dis"] for collision in self.collision_all])
        fig2, axes = FigFon.get_figure_axes(SubFigSize=(1, 1))
        axes.plot(t, col_dis)
        axes.set_xlabel("t/s")
        axes.set_ylabel("closest distance/m")
        plt.legend([str(i) for i in range(self.model.env.num_envs)])
        plt.show()

        fig3, axes3 = FigFon.get_figure_axes(SubFigSize=(1, 1))

        axes3.plot(t[1:], self.reward_all.T)
        axes3.set_xlabel("t/s")
        axes3.set_ylabel("reward")
        plt.legend([str(i) for i in range(self.model.env.num_envs)])

        plt.show()

        return [fig, fig2]
