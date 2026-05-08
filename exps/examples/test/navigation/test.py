from typing import Optional

import numpy as np
import torch as th
from matplotlib import pyplot as plt

from VisFly.utils.FigFashion.FigFashion import FigFon
from VisFly.utils.evaluate import TestBase


class Test(TestBase):
    def __init__(
            self,
            model,
            name,
            save_path: Optional[str] = None,
    ):
        super(Test, self).__init__(model=model, name=name, save_path=save_path)

    def draw(self, names=None):
        state_data = th.as_tensor(np.asarray([obs["state"] for obs in self.obs_all])).cpu().numpy()
        t = th.stack([th.as_tensor(item).reshape(-1)[0] for item in self.t]).cpu().numpy()
        figs = []

        for i in range(self.model.env.num_envs):
            fig = plt.figure(figsize=(5, 4))
            figs.append(fig)

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

        col_dis = th.stack([th.as_tensor(collision["col_dis"]) for collision in self.collision_all]).cpu().numpy()
        fig2, axes = FigFon.get_figure_axes(SubFigSize=(1, 1))
        axes.plot(t, col_dis)
        axes.set_xlabel("t/s")
        axes.set_ylabel("closest distance/m")
        figs.append(fig2)

        return figs
