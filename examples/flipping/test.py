import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional
import os, sys
import torch
from VisFly.utils.evaluate import TestBase
from VisFly.utils.FigFashion.FigFashion import FigFon
from VisFly.utils.maths import Quaternion

class Test(TestBase):
    def __init__(self, model, name, save_path: Optional[str] = None):
        super(Test, self).__init__(env=None, model=model, name=name, save_path=save_path)

    def draw(self, names=None):
        # collect state history: position, quaternion, velocity, and angular velocity
        state_data = [obs["state"] for obs in self.obs_all]
        state_data = np.array(state_data)
        # time steps
        t = np.array(self.t)[:, 0]
        # Create figure with 2x2 subplot layout
        fig = plt.figure(figsize=(16, 12))
        # Position plot (2D time series)
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(t, state_data[:, 0, 0:3])
        ax1.set_title("Position vs Time")
        ax1.set_ylabel("Position (m)")
        ax1.set_xlabel("Time (s)")
        ax1.legend(["x", "y", "z"])
        ax1.grid(True)
        # 3D Trajectory plot
        ax2 = plt.subplot(2, 2, 2, projection='3d')
        x_pos = state_data[:, 0, 0]
        y_pos = state_data[:, 0, 1]
        z_pos = state_data[:, 0, 2]
        ax2.plot(x_pos, y_pos, z_pos, 'b-', linewidth=2, alpha=0.7, label='Trajectory')
        ax2.scatter(x_pos[0], y_pos[0], z_pos[0], color='green', s=100, label='Start', marker='o')
        ax2.scatter(x_pos[-1], y_pos[-1], z_pos[-1], color='red', s=100, label='End', marker='s')
        n_points = min(50, len(x_pos))
        indices = np.linspace(0, len(x_pos)-1, n_points).astype(int)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))
        for i, idx in enumerate(indices):
            ax2.scatter(x_pos[idx], y_pos[idx], z_pos[idx], color=colors[i], s=20, alpha=0.6)
        ax2.set_xlabel("X Position (m)")
        ax2.set_ylabel("Y Position (m)")
        ax2.set_zlabel("Z Position (m)")
        ax2.set_title("3D Trajectory")
        ax2.legend()
        max_range = np.array([x_pos.max()-x_pos.min(), y_pos.max()-y_pos.min(), z_pos.max()-z_pos.min()]).max() / 2.0
        mid_x = (x_pos.max()+x_pos.min()) * 0.5
        mid_y = (y_pos.max()+y_pos.min()) * 0.5
        mid_z = (z_pos.max()+z_pos.min()) * 0.5
        ax2.set_xlim(mid_x - max_range, mid_x + max_range)
        ax2.set_ylim(mid_y - max_range, mid_y + max_range)
        ax2.set_zlim(mid_z - max_range, mid_z + max_range)
        # Orientation quaternion plot
        ax3 = plt.subplot(2, 2, 3)
        state_data = torch.from_numpy(state_data)
        q = Quaternion(state_data[:, 0, 3], state_data[:, 0, 4], state_data[:, 0, 5], state_data[:, 0, 6]).toEuler().T
        # to angle
        angles = q*57.2958
        ax3.plot(t, angles)
        ax3.set_title("Orientation Euler")
        ax3.set_ylabel("Euler")
        ax3.set_xlabel("Time (s)")
        ax3.legend(["roll", "pitch", "yaw"])
        ax3.grid(True)
        # Angular velocity plot
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(t, state_data[:, 0, 10:13])
        ax4.set_title("Angular Velocity")
        ax4.set_ylabel("Angular Velocity (rad/s)")
        ax4.set_xlabel("Time (s)")
        ax4.legend(["wx", "wy", "wz"])
        ax4.grid(True)
        plt.tight_layout()
        plt.show()
        return [fig] 