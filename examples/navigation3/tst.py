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

    def create_combined_video_frame(self, global_frame, agent_sensor_data, timestep_idx):
        """
        Create a combined frame with global view and agent sensor views.
        
        Args:
            global_frame: Global camera rendered frame
            agent_sensor_data: Dictionary containing sensor data for all agents
            timestep_idx: Current timestep index
        
        Returns:
            Combined frame as numpy array
        """
        # Get global frame dimensions
        global_h, global_w = global_frame.shape[:2]
        
        # Collect agent sensor frames (depth images)
        agent_frames = []
        if "depth" in agent_sensor_data:
            depth_data = agent_sensor_data["depth"]
            
            # Convert to CPU and numpy if it's a CUDA tensor
            if hasattr(depth_data, 'cpu'):
                depth_data = depth_data.cpu().numpy()
            elif hasattr(depth_data, 'numpy'):
                depth_data = depth_data.numpy()
            
            # Normalize and convert depth to RGB for visualization
            for agent_idx in range(depth_data.shape[0]):
                depth_frame = depth_data[agent_idx, 0]  # Remove channel dimension
                # Normalize depth to 0-255 range
                depth_max = depth_frame.max()
                if depth_max > 0:
                    depth_normalized = ((depth_frame / depth_max) * 255).astype(np.uint8)
                else:
                    depth_normalized = np.zeros_like(depth_frame, dtype=np.uint8)
                # Convert to RGB
                depth_rgb = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
                # Resize to reasonable size
                agent_frames.append(cv2.resize(depth_rgb, (200, 200)))
        
        # Create layout: global view on left, agent views on right
        if agent_frames:
            # Calculate grid layout for agent views
            num_agents = len(agent_frames)
            grid_cols = min(4, num_agents)  # Max 4 columns
            grid_rows = (num_agents + grid_cols - 1) // grid_cols
            
            # Create agent grid
            agent_grid_h = grid_rows * 200
            agent_grid_w = grid_cols * 200
            agent_grid = np.zeros((agent_grid_h, agent_grid_w, 3), dtype=np.uint8)
            
            for i, frame in enumerate(agent_frames):
                row = i // grid_cols
                col = i % grid_cols
                y_start = row * 200
                x_start = col * 200
                agent_grid[y_start:y_start+200, x_start:x_start+200] = frame
            
            # Resize global frame to match agent grid height
            global_resized = cv2.resize(global_frame, (int(global_w * agent_grid_h / global_h), agent_grid_h))
            
            # Combine horizontally
            combined_frame = np.hstack([global_resized, agent_grid])
        else:
            combined_frame = global_frame
        
        # Add text overlay with timestep info
        cv2.putText(combined_frame, f"Timestep: {timestep_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return combined_frame

    def save_combined_video(self, episode_dir):
        """
        Save a combined video with global camera view and agent sensor views.
        """
        if not self.render_image_all:
            print("No render images available for video creation")
            return
        
        print(f"Creating combined video for episode...")
        
        # Create combined frames
        combined_frames = []
        for i, (render_frame, obs_data) in enumerate(zip(self.render_image_all, self.obs_all)):
            # Convert render frame from RGB to BGR for OpenCV
            render_bgr = cv2.cvtColor(render_frame, cv2.COLOR_RGB2BGR)
            
            # Create combined frame
            combined_frame = self.create_combined_video_frame(render_bgr, obs_data, i)
            combined_frames.append(combined_frame)
        
        if not combined_frames:
            print("No frames to save")
            return
        
        # Video settings
        height, width = combined_frames[0].shape[:2]
        fps = int(1.0 / self.env.envs.dynamics.dt)  # Use environment timestep
        
        # Save combined video
        video_path = os.path.join(episode_dir, "combined_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        for frame in combined_frames:
            video_writer.write(frame)
        
        video_writer.release()
        print(f"Combined video saved: {video_path}")
        
        # Also save individual videos
        self.save_individual_videos(episode_dir)

    def save_individual_videos(self, episode_dir):
        """
        Save individual videos for global view and each agent's sensor data.
        """
        if not self.render_image_all:
            return
        
        # Save global camera video
        self.save_global_video(episode_dir)
        
        # Save agent depth videos
        self.save_agent_depth_videos(episode_dir)

    def save_global_video(self, episode_dir):
        """Save global camera view video."""
        height, width, _ = self.render_image_all[0].shape
        fps = int(1.0 / self.env.envs.dynamics.dt)
        
        video_path = os.path.join(episode_dir, "global_camera.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        for frame in self.render_image_all:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
        print(f"Global camera video saved: {video_path}")

    def save_agent_depth_videos(self, episode_dir):
        """Save individual agent depth sensor videos."""
        if not self.obs_all or "depth" not in self.obs_all[0]:
            return
        
        num_agents = self.obs_all[0]["depth"].shape[0]
        fps = int(1.0 / self.env.envs.dynamics.dt)
        
        # Create videos for each agent
        agent_writers = {}
        
        for agent_idx in range(num_agents):
            video_path = os.path.join(episode_dir, f"agent_{agent_idx}_depth.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            agent_writers[agent_idx] = cv2.VideoWriter(video_path, fourcc, fps, (200, 200))
        
        # Process each timestep
        for obs_data in self.obs_all:
            if "depth" in obs_data:
                depth_data = obs_data["depth"]
                for agent_idx in range(num_agents):
                    if agent_idx < depth_data.shape[0]:
                        # Get depth frame for this agent
                        depth_frame = depth_data[agent_idx, 0]  # Remove channel dimension
                        
                        # Convert depth to numpy if it's a tensor
                        if hasattr(depth_frame, 'cpu'):
                            depth_frame = depth_frame.cpu().numpy()
                        elif hasattr(depth_frame, 'numpy'):
                            depth_frame = depth_frame.numpy()
                        
                        # Normalize and colorize depth
                        depth_normalized = ((depth_frame / (depth_frame.max() + 1e-6)) * 255).astype(np.uint8)
                        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
                        depth_resized = cv2.resize(depth_colored, (200, 200))
                        
                        agent_writers[agent_idx].write(depth_resized)
        
        # Close all video writers
        for agent_idx, writer in agent_writers.items():
            writer.release()
            print(f"Agent {agent_idx} depth video saved: agent_{agent_idx}_depth.mp4") 

    def draw(self, names=None):
        # Use actual state data which contains absolute positions
        state_data = [state for state in self.state_all]
        # Move tensors to CPU before converting to numpy
        state_data = [state.cpu() if hasattr(state, 'cpu') else state for state in state_data]
        state_data = np.array(state_data)
        
        # Extract absolute positions from state data (first 3 columns are XYZ positions)
        absolute_positions = state_data[:, :, 0:3]
        
        # Get target positions for plotting
        targets = self.obs_all[0]["target"]
        if hasattr(targets, 'cpu'):
            targets = targets.cpu().numpy()
        
        # Get number of environments from state data
        num_envs = state_data.shape[1]
        
        # Determine which agents succeeded at any point during the episode
        success_agents = set()
        for timestep_info in self.info_all:
            for agent_idx, agent_info in enumerate(timestep_info):
                if agent_info and "is_success" in agent_info and agent_info["is_success"]:
                    success_agents.add(agent_idx)
        
        # time steps - handle CUDA tensors properly
        t_list = []
        for time_tensor in self.t:
            if hasattr(time_tensor, 'cpu'):
                t_list.append(time_tensor.cpu().numpy())
            else:
                t_list.append(np.array(time_tensor))
        t = np.array(t_list)[:, 0]
        
        # Create figure with 2x2 subplot layout
        fig = plt.figure(figsize=(16, 12))
        
        # ------- Helper: draw many agents faint + mean bold ------------------
        def _plot_with_mean(ax, series, labels, title, ylabel, fill_alpha=0.15):
            """Plot mean curve with shaded ±1 σ band to avoid clutter from many agents."""
            if series.ndim == 2:  # (time, channels)
                series = series[:, None, :]

            mean_vals = series.mean(axis=1)
            std_vals = series.std(axis=1)

            for ch in range(series.shape[2]):
                ax.plot(t, mean_vals[:, ch], linewidth=2, label=labels[ch])
                ax.fill_between(
                    t,
                    mean_vals[:, ch] - std_vals[:, ch],
                    mean_vals[:, ch] + std_vals[:, ch],
                    alpha=fill_alpha,
                )

            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Time (s)")
            ax.legend()
            ax.grid(True)

        # Position plot (x,y,z) - use absolute positions from state data
        ax1 = plt.subplot(2, 2, 1)
        _plot_with_mean(ax1, absolute_positions, ["x", "y", "z"], "Position vs Time", "Position (m)")
        
        # 3D Trajectory plot – plot every agent/roll-out in the same axis
        ax2 = plt.subplot(2, 2, 2, projection='3d')

        # Define a colormap to give each agent a distinct hue for orientation / markers
        cmap = plt.cm.get_cmap('tab20', num_envs)

        # Loop over each agent/roll-out and plot its trajectory
        scatter_handles = []
        for agent_idx in range(num_envs):
            # Use absolute positions from state data
            x_pos = absolute_positions[:, agent_idx, 0]
            y_pos = absolute_positions[:, agent_idx, 1]
            z_pos = absolute_positions[:, agent_idx, 2]

            # Compute speed magnitude from velocity components (indices 7:10 in state)
            vel = state_data[:, agent_idx, 7:10]
            speed = np.linalg.norm(vel, axis=1)

            # Plot grey line for trajectory path (outline)
            ax2.plot(x_pos, y_pos, z_pos, color='lightgrey', linewidth=0.7, alpha=0.6)

            # Scatter points colored by speed
            sc = ax2.scatter(x_pos, y_pos, z_pos, c=speed, cmap='viridis', s=8, alpha=0.9)

            # Start marker (green circle) - now using correct absolute position
            ax2.scatter(x_pos[0], y_pos[0], z_pos[0], color='green', s=50, marker='o', edgecolor='black', linewidth=1)

            # End markers with success/failure indication
            if agent_idx in success_agents:
                # Blue square for successful agents
                ax2.scatter(x_pos[-1], y_pos[-1], z_pos[-1], color='blue', s=50, marker='s', edgecolor='black', linewidth=1)
            else:
                # Red X for failed agents (remove edgecolor to avoid warning)
                ax2.scatter(x_pos[-1], y_pos[-1], z_pos[-1], color='red', s=50, marker='x', linewidth=1)

        # Add target positions for all agents (if available in observations)
        if "target" in self.obs_all[0]:
            # Plot target positions as gold stars
            for agent_idx in range(num_envs):
                # Handle different target array shapes
                if len(targets.shape) == 1:  # Single agent case: targets is 1D array [x, y, z]
                    if agent_idx == 0 and len(targets) >= 3:  # Only plot for the first (and only) agent
                        ax2.scatter(targets[0], targets[1], targets[2], 
                                   color='gold', s=80, marker='*', edgecolor='black', linewidth=1, 
                                   label='Target')
                elif len(targets.shape) == 2:  # Multi-agent case or single agent stored as 2D: targets is 2D array
                    if targets.shape[0] > agent_idx:  # Check if this agent index exists
                        target_pos = targets[agent_idx]
                        if len(target_pos) >= 3:  # ensure 3D coordinates
                            ax2.scatter(target_pos[0], target_pos[1], target_pos[2], 
                                       color='gold', s=80, marker='*', edgecolor='black', linewidth=1, 
                                       label='Target' if agent_idx == 0 else None)  # label only once
        else:
            print(f"DEBUG: No target found in observations")
        
        # Additional direct target plotting to ensure visibility
        if "target" in self.obs_all[0]:
            print(f"DEBUG: Target found in observations")
            print(f"DEBUG: targets shape: {targets.shape}")
            print(f"DEBUG: targets content: {targets}")
            print(f"DEBUG: num_envs: {num_envs}")
            
            # Force plot target for all agents with a simpler approach
            if len(targets.shape) == 1 and len(targets) >= 3:
                print(f"DEBUG: Plotting single agent target at: {targets[0]}, {targets[1]}, {targets[2]}")
                # Single agent case
                ax2.scatter(targets[0], targets[1], targets[2], 
                           color='gold', s=120, marker='*', edgecolor='red', linewidth=2, 
                           label='Target (Direct)', zorder=10)  # High zorder to be on top
            elif len(targets.shape) == 2:
                print(f"DEBUG: Plotting multi-agent targets")
                # Multi-agent case
                for i in range(min(targets.shape[0], num_envs)):
                    if len(targets[i]) >= 3:
                        print(f"DEBUG: Agent {i} target at: {targets[i][0]}, {targets[i][1]}, {targets[i][2]}")
                        ax2.scatter(targets[i][0], targets[i][1], targets[i][2], 
                                   color='gold', s=120, marker='*', edgecolor='red', linewidth=2, 
                                   label='Target (Direct)' if i == 0 else None, zorder=10)
            else:
                print(f"DEBUG: Unexpected targets shape: {targets.shape}")
        else:
            print(f"DEBUG: No target found in observations keys: {list(self.obs_all[0].keys())}")
        
        ax2.set_xlabel("X Position (m)")
        ax2.set_ylabel("Y Position (m)")
        ax2.set_zlabel("Z Position (m)")
        ax2.set_title("3D Trajectory")
        
        # Make the 3D plot aspect ratio equal - use absolute positions and include targets
        all_x = absolute_positions[:, :, 0].flatten()
        all_y = absolute_positions[:, :, 1].flatten()
        all_z = absolute_positions[:, :, 2].flatten()
        
        # Include target positions in the bounds calculation
        if "target" in self.obs_all[0]:
            if len(targets.shape) == 1 and len(targets) >= 3:
                all_x = np.append(all_x, targets[0])
                all_y = np.append(all_y, targets[1])
                all_z = np.append(all_z, targets[2])
            elif len(targets.shape) == 2:
                for i in range(min(targets.shape[0], num_envs)):
                    if len(targets[i]) >= 3:
                        all_x = np.append(all_x, targets[i][0])
                        all_y = np.append(all_y, targets[i][1])
                        all_z = np.append(all_z, targets[i][2])

        max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
        mid_x = (all_x.max()+all_x.min()) * 0.5
        mid_y = (all_y.max()+all_y.min()) * 0.5
        mid_z = (all_z.max()+all_z.min()) * 0.5
        ax2.set_xlim(mid_x - max_range, mid_x + max_range)
        ax2.set_ylim(mid_y - max_range, mid_y + max_range)
        ax2.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Orientation quaternion plot (assuming indices 3:7 are quaternion)
        ax3 = plt.subplot(2, 2, 3)
        _plot_with_mean(ax3, state_data[:, :, 3:7], ["w", "x", "y", "z"], "Orientation Quaternion", "Quaternion")
        
        # Angular velocity plot (indices 10:13 in state)
        ax4 = plt.subplot(2, 2, 4)
        _plot_with_mean(ax4, state_data[:, :, 10:13], ["wx", "wy", "wz"], "Angular Velocity", "Angular Velocity (rad/s)")
        
        plt.tight_layout()
        plt.show()
        return [fig] 