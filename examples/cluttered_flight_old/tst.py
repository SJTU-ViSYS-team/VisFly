import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional
import os, sys
import cv2

from VisFly.utils.evaluate import TestBase
from VisFly.utils.FigFashion.FigFashion import FigFon

class Test(TestBase):
    def __init__(self, env, model, name, save_path: Optional[str] = None):
        super(Test, self).__init__(env=env, model=model, name=name, save_path=save_path)

    # ------------- Advanced plotting & video utilities from navigation3 ----------------
    def create_combined_video_frame(self, global_frame, agent_sensor_data, timestep_idx):
        """Compose a frame that shows the global camera plus depth views of all agents."""
        global_h, global_w = global_frame.shape[:2]

        # --- build per-agent depth frames -------------------------------------------------
        agent_frames = []
        if "depth" in agent_sensor_data:
            depth = agent_sensor_data["depth"]
            depth = depth.cpu().numpy() if hasattr(depth, "cpu") else depth.numpy() if hasattr(depth, "numpy") else depth
            for d in depth[:, 0]:
                dmax = d.max() if d.max() > 0 else 1.0
                depth_norm = ((d / dmax) * 255).astype(np.uint8)
                agent_frames.append(cv2.applyColorMap(depth_norm, cv2.COLORMAP_VIRIDIS))

        # --- layout global + agent grid ---------------------------------------------------
        if agent_frames:
            cols = min(4, len(agent_frames))
            rows = (len(agent_frames) + cols - 1) // cols
            grid = np.zeros((rows * 200, cols * 200, 3), dtype=np.uint8)
            for idx, frame in enumerate(agent_frames):
                r, c = divmod(idx, cols)
                grid[r*200:(r+1)*200, c*200:(c+1)*200] = cv2.resize(frame, (200, 200))
            global_resized = cv2.resize(global_frame, (int(global_w * grid.shape[0] / global_h), grid.shape[0]))
            combined = np.hstack([global_resized, grid])
        else:
            combined = global_frame

        cv2.putText(combined, f"Timestep: {timestep_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return combined

    def save_combined_video(self, episode_dir):
        if not self.render_image_all:
            print("No render images available for video creation")
            return
        frames = [self.create_combined_video_frame(cv2.cvtColor(r, cv2.COLOR_RGB2BGR), o, i)
                  for i, (r, o) in enumerate(zip(self.render_image_all, self.obs_all))]
        if not frames:
            return
        h, w = frames[0].shape[:2]
        fps = int(1.0 / self.env.envs.dynamics.dt)
        path = os.path.join(episode_dir, "combined_video.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        print(f"Combined video saved: {path}")
        self.save_individual_videos(episode_dir)

    def save_individual_videos(self, episode_dir):
        if not self.render_image_all:
            return
        self.save_global_video(episode_dir)
        self.save_agent_depth_videos(episode_dir)

    def save_global_video(self, episode_dir):
        h, w, _ = self.render_image_all[0].shape
        fps = int(1.0 / self.env.envs.dynamics.dt)
        path = os.path.join(episode_dir, "global_camera.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for img in self.render_image_all:
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Global camera video saved: {path}")

    def save_agent_depth_videos(self, episode_dir):
        if not self.obs_all or "depth" not in self.obs_all[0]:
            return
        depth_seq = [o["depth"].cpu().numpy() if hasattr(o["depth"], "cpu") else o["depth"].numpy() for o in self.obs_all]
        num_agents = depth_seq[0].shape[0]
        fps = int(1.0 / self.env.envs.dynamics.dt)
        writers = {i: cv2.VideoWriter(os.path.join(episode_dir, f"agent_{i}_depth.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (200, 200)) for i in range(num_agents)}
        for depth in depth_seq:
            for i in range(num_agents):
                d = depth[i, 0]
                dn = ((d / (d.max() + 1e-6)) * 255).astype(np.uint8)
                writers[i].write(cv2.resize(cv2.applyColorMap(dn, cv2.COLORMAP_VIRIDIS), (200, 200)))
        for i, w in writers.items():
            w.release()
            print(f"Agent {i} depth video saved: agent_{i}_depth.mp4")

    # ------------------------ Enhanced draw (mean + shaded var) -------------------------
    def draw(self, names=None):
        import matplotlib.pyplot as plt, numpy as np
        state_data = np.array([s.cpu() if hasattr(s, 'cpu') else s for s in self.state_all])
        abs_pos = state_data[:, :, 0:3]
        targets = self.obs_all[0].get("target", None)
        if targets is not None and hasattr(targets, 'cpu'):
            targets = targets.cpu().numpy()
        num_envs = state_data.shape[1]
        success_agents = {idx for step in self.info_all for idx, info in enumerate(step) if info and info.get("is_success", False)}
        t = np.array([ti.cpu().numpy() if hasattr(ti,'cpu') else ti for ti in self.t])[:, 0]

        fig = plt.figure(figsize=(16, 12))

        def _plot(ax, series, labels, title, ylabel):
            if series.ndim == 2:
                series = series[:, None, :]
            mean, std = series.mean(1), series.std(1)
            for ch in range(series.shape[2]):
                ax.plot(t, mean[:, ch], label=labels[ch], linewidth=2)
                ax.fill_between(t, mean[:, ch]-std[:, ch], mean[:, ch]+std[:, ch], alpha=0.15)
            ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel("Time (s)"); ax.legend(); ax.grid(True)

        _plot(plt.subplot(2,2,1), abs_pos, ["x","y","z"], "Position vs Time", "Position (m)")

        ax2 = plt.subplot(2,2,2, projection='3d')
        for i in range(num_envs):
            x, y, z = abs_pos[:, i, 0], abs_pos[:, i, 1], abs_pos[:, i, 2]
            vel = state_data[:, i, 7:10]
            speed = np.linalg.norm(vel, axis=1)
            ax2.plot(x, y, z, color='lightgrey', linewidth=0.7, alpha=0.6)
            ax2.scatter(x, y, z, c=speed, cmap='viridis', s=8, alpha=0.9)
            ax2.scatter(x[0], y[0], z[0], color='green', s=50, marker='o', edgecolor='black')
            end_color = 'blue' if i in success_agents else 'red'
            end_marker = 's' if i in success_agents else 'x'
            ax2.scatter(x[-1], y[-1], z[-1], color=end_color, s=50, marker=end_marker, edgecolor='black' if i in success_agents else None)
        if targets is not None:
            if targets.ndim == 1 and targets.size >= 3:
                ax2.scatter(*targets[:3], color='gold', s=120, marker='*', edgecolor='red', linewidth=2)
            elif targets.ndim == 2:
                for i in range(min(targets.shape[0], num_envs)):
                    ax2.scatter(*targets[i, :3], color='gold', s=120, marker='*', edgecolor='red', linewidth=2)
        all_x, all_y, all_z = abs_pos[:, :, 0].flatten(), abs_pos[:, :, 1].flatten(), abs_pos[:, :, 2].flatten()
        if targets is not None:
            all_x, all_y, all_z = np.append(all_x, targets[...,0]), np.append(all_y, targets[...,1]), np.append(all_z, targets[...,2])
        mr = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max()/2
        mid = (all_x.max()+all_x.min())/2, (all_y.max()+all_y.min())/2, (all_z.max()+all_z.min())/2
        ax2.set_xlim(mid[0]-mr, mid[0]+mr); ax2.set_ylim(mid[1]-mr, mid[1]+mr); ax2.set_zlim(mid[2]-mr, mid[2]+mr)
        ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z'); ax2.set_title('3D Trajectory')

        _plot(plt.subplot(2,2,3), state_data[:,:,3:7], ["w","x","y","z"], "Orientation Quaternion", "Quaternion")
        _plot(plt.subplot(2,2,4), state_data[:,:,10:13], ["wx","wy","wz"], "Angular Velocity", "Angular Velocity (rad/s)")

        plt.tight_layout(); plt.show()
        return [fig] 