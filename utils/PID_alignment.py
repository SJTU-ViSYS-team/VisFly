import rosbag
import numpy as np
import os
import sys
import torch as th
import matplotlib.pyplot as plot
import json
import time
sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as R_scipy
from std_msgs.msg import Float32MultiArray  # 假设动作数据是 Float32MultiArray 类型
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from sensor_msgs.msg import Imu
from torch.nn import functional as F
from scipy import interpolate
from scipy.spatial.transform import Slerp
from VisFly.utils.type import bound
from VisFly.utils.FigFashion.colors import colorsets
from VisFly.envs.base.dynamics import Dynamics
from VisFly.envs.HoverEnv import HoverEnv
MODE_CHANNEL = 6 
HOVER_ACC = 9.81 
MODE_SHIFT_VALUE = 0.25
colors = colorsets["Modern Scientific"]

cfg = "drone_state"
class bag_plot:
    def __init__(self):
        
        # self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.scene_path = "datasets/spy_datasets/configs/garage_empty"
        self.m = 0.46
        # self.bag_file = '/home/suncc/suncc/My-research/Vision-based-Racing/waypoint1.bag'
        # self.bag_file = 'debug/pid_test_all_y2.bag'
        # self.bag_file = 'debug/pid_test_all_z2.bag'
        self.bag_file = 'debug/pid_test_all.bag'
        self.bag = rosbag.Bag(self.bag_file)
        self.topics = ['/bfctrl/cmd', '/mavros/setpoint_raw/attitude', '/bfctrl/local_odom', '/mavros/imu/data']
        self.actions_cmd = []
        self.actions_real = []
        self.action_drone = []
        self.anglevel = []
        self.thrust = []
        self.reward_all = []
        self.action_all = []
        self.state_all = []
        self.state_bf = []
        self.state_bf_time = []
        self.obs_all = []
        self.position_all = []
        self.actions_time = []  
        self.ctrli = []
        self.action_drone_time = []  
        self.imu_angle_v = []
        self.imu_time = []
        
        self.env= HoverEnv(num_agent_per_scene=1,
                            # num_scene=1,
                            visual=False, # 不用视觉要改成False
                            max_episode_steps=512,
                            scene_kwargs={
                                "path": self.scene_path,
                            },
                            dynamics_kwargs={
                                "cfg":cfg
                            }
                            )

        self.bag_parser()
        self.align_data()
        self.load(f"VisFly/configs/{cfg}.json")
        self.simulator_actions()
        self.plot()
        
    def bag_parser(self):
        for topic, msg, t in self.bag.read_messages(self.topics):  

            if topic == '/bfctrl/cmd':
                ctbr_x , ctbr_y, ctbr_z, thrust = msg.angularVel.x, msg.angularVel.y, msg.angularVel.z, msg.thrust
                self.actions_cmd.append(np.array([thrust, ctbr_x, ctbr_y, ctbr_z], dtype=np.float32))  # 假设动作数据存储在 msg.data 中
                self.actions_time.append(t.to_sec())  
                
            if topic == '/mavros/setpoint_raw/attitude':
                thrust, anglevel_x, anglevel_y, anglevel_z = msg.thrust, msg.body_rate.x, msg.body_rate.y, msg.body_rate.z
                self.action_drone.append(np.array([thrust, anglevel_x, anglevel_y, anglevel_z], dtype=np.float32))
                self.action_drone_time.append(t.to_sec())
                
            if topic =='/bfctrl/local_odom':
                pos_x, pos_y, pos_z, ori_w, ori_x, ori_y, ori_z = msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z
                self.state_bf.append(np.array([pos_x, pos_y, pos_z, ori_w, ori_x, ori_y, ori_z], dtype=np.float32))
                self.state_bf_time.append(t.to_sec()) 
            
            if topic == '/mavros/imu/data':
                angle_x, angle_y, angle_z = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
                self.imu_angle_v.append(np.array([angle_x, angle_y, angle_z], dtype=np.float32))
                self.imu_time.append(t.to_sec())
                
        self.actions_real = th.as_tensor(self.actions_cmd.copy())
        self.state_bf = th.as_tensor(self.state_bf)
        self.action_drone = th.as_tensor(self.action_drone)
        self.actions_time = np.array(self.actions_time)
        self.action_drone_time = np.array(self.action_drone_time)
        # self.actions_real = th.as_tensor(self.actions.copy())
        
        self.actions_role = self.actions_cmd
                    
    def load(self, path=""):
        with open(path, "r") as f:
            data = json.load(f)
        self._bd_rate = bound(
            max=th.tensor(data["max_rate"]), min=th.tensor(-data["max_rate"])
        )
        
    def normalize(self, command, normal_range: [float, float] = (-1, 1)):
        # thrust
        thrust_scale = 1 * HOVER_ACC
        normalized_thrust = (command[:, :1] - 1 * HOVER_ACC) / thrust_scale

        # anglevel
        bodyrate_scale = (self._bd_rate.max - self._bd_rate.min) / (normal_range[1] - normal_range[0])
        bodyrate_bias = self._bd_rate.max - bodyrate_scale * normal_range[1]
        normalized_bodyrate = (command[:, 1:] - bodyrate_bias) / bodyrate_scale

        normalized_command = th.hstack([normalized_thrust, normalized_bodyrate])
        return normalized_command
    
    def align_data(self):
        self.start_time, self.end_time = self.actions_time[0], self.actions_time[-1]
        def get_cut_data(ori_data, ori_time, start_time, end_time):
            mask = (ori_time >= start_time) & (ori_time <= end_time)
            return ori_data[mask], np.array(ori_time)[mask]
        self.ori_action, self.ori_action_time = get_cut_data(
            self.actions_real, self.actions_time, self.start_time, self.end_time)
        
        self.imu_angle_v, self.imu_time = get_cut_data(
            np.array(self.imu_angle_v), self.imu_time, self.start_time, self.end_time)
        
        self.ori_action_time -= self.start_time
        self.imu_time -= self.start_time
        
        act_f = [interpolate.interp1d(
            self.ori_action_time, self.ori_action[:,i],
            kind='nearest',
            fill_value=(self.ori_action[0,i], self.ori_action[-1,i]),
            bounds_error=False
        ) for i in range(self.ori_action.shape[1])]
        
        self.act_f = lambda t:  np.array([f(t) for f in act_f]).T
        
        # self.actions_aligned = self._interpolate_data(
        #     self.actions_time, self.actions_real.numpy(), 4)
        
        # self.action_drone_aligned = self._interpolate_data(
        #     self.action_drone_time, self.action_drone.numpy(), 4)
        
        # # 对齐位置数据
        # state_bf_pos = self.state_bf.numpy()[:,:3]
        # self.state_bf_pos_aligned = self._interpolate_data(
        #     self.state_bf_time, state_bf_pos, 3)
        
        # # 对齐姿态数据（四元数球面插值）
        # quaternions = self.state_bf.numpy()[:,3:7]
        # rotations = R_scipy.from_quat(quaternions[:,[1,2,3,0]])
        # slerp = Slerp(self.state_bf_time, rotations)
        # self.state_bf_ori_aligned = np.array([
        #     slerp(t).as_quat()[[3,0,1,2]] for t in self.base_time
        # ])
        
        # 对齐IMU数据
        # self.imu_aligned = self._interpolate_data(
        #     self.imu_time, np.array(self.imu_angle_v), 3)
            
        # # 时间标准化（相对时间）
        # self.time_offset = self.base_time[0]
        # self.base_time -= self.time_offset
        
    def simulator_actions(self):
        obs = self.env.reset()
        self.action_role = self.normalize(th.as_tensor(self.act_f(self.env.envs.dynamics.t)))
        self.sim_t = []
        while self.env.envs.dynamics.t < self.end_time-self.start_time:
            action= self.normalize(th.as_tensor(self.act_f(self.env.envs.dynamics.t)))
            action[:,0] = 0
            # action = action.unsqueeze(0)
            print(f"Action: {action}")
            self.sim_t.append(self.env.envs.dynamics.t.clone().detach())
            # obs, reward, done, info = self.env.step(action)
            action = th.as_tensor(action, dtype=th.float32)
            obs = self.env.envs.dynamics.step(action) 
            self.obs_all.append(obs)
            # self.reward_all.append(reward)
            self.action_all.append(action)
            self.state_all.append(self.env.state)
            print(self.env.state[0,10:])
            self.anglevel.append(self.env.angular_velocity.clone().detach())
            # self.ctrli.append(self.env.envs.dynamics._ctrl_i)
            # self.thrust.append(self.env.thrust)

            # if done:
            #     obs = self.env.reset()  # 如果 episode 结束，重置环境
                
        self.anglevel = th.cat(self.anglevel, dim=0)
        # self.thrust = th.cat(self.thrust, dim=0)
        # self.reward_all = th.tensor(self.reward_all)
        self.action_all = th.cat(self.action_all, dim=0)
        self.state_all = th.cat(self.state_all, dim=0)
        # self.ctrli = th.cat(self.ctrli, dim=1)
        self.sim_t = th.cat(self.sim_t)
    
    def plot(self):
        # 执行时间戳对齐
        # self.align_data()
        # fig_ctbr1, axs = plot.subplots(figsize=(35, 10), nrows=3, ncols=1)
        self.imu_angle_v = np.stack(self.imu_angle_v)
        
        # for i, ax in enumerate(axs):
        # ##############################################################PID_angle_v
        #     # fig_ctbr, ax = plot.subplots(2, 2, figsize=(25, 15))
        #     # fig_ctbr1, ax = plot.subplots(figsize=(35, 10))
        #     # ax.plot(self.base_time[1120:1500], self.actions_aligned[1120:1500, 1], label="anglex_cmd_real", color=colors[0])
        #     # ax.plot(self.base_time[1120:1500], self.anglevel[1120:1500, 0].numpy(), label="anglex_sim", color=colors[1])
        #     # ax.plot(self.base_time[1120:1500], self.imu_aligned[1120:1500,0], label="anglex_imu_real", color=colors[2])
        #     # ax.set_title("anglevel_x")
        #     # ax.legend()
        #     ax.plot(self.ori_action_time, self.ori_action[:,i+1], label="cmd_real", color=colors[0])
        #     ax.plot(self.sim_t, self.action_all.numpy()[:,i+1], label="cmd_sim", color=colors[3])
        #     ax.plot(self.imu_time, self.imu_angle_v[:,i], label="anglex_real", color=colors[2])
        #     ax.plot(self.sim_t, self.anglevel.numpy()[:,i], label="anglex_sim",color=colors[1])
        #     # index to x.y.z string
            
        #     ax.set_title(f"anglevel_{i+1}")
        #     ax.legend()
            
        fig, axs = plot.subplots(figsize=(35, 10), nrows=1, ncols=1)
        axs.plot(self.ori_action_time, self.ori_action[:,1], label="cmd_real", color=colors[0])
        axs.plot(self.sim_t, self.action_all.numpy()[:,1], label="cmd_sim", color=colors[3])
        axs.plot(self.imu_time, self.imu_angle_v[:,0], label="anglex_real", color=colors[2])
        axs.plot(self.sim_t, self.anglevel.numpy()[:,0], label="anglex_sim",color=colors[1])
        axs.set_title("anglevel_x")
        # index to x.y.z string
        figy, axsy = plot.subplots(figsize=(35, 10), nrows=1, ncols=1)
        axsy.plot(self.ori_action_time, self.ori_action[:,2], label="cmd_real", color=colors[0])
        axsy.plot(self.sim_t, self.action_all.numpy()[:,2], label="cmd_sim", color=colors[3])
        axsy.plot(self.imu_time, self.imu_angle_v[:,1], label="anglex_real", color=colors[2])
        axsy.plot(self.sim_t, self.anglevel.numpy()[:,1], label="anglex_sim",color=colors[1])
        axsy.set_title("anglevel_y")
        
        figz, axsz = plot.subplots(figsize=(35, 10), nrows=1, ncols=1)
        axsz.plot(self.ori_action_time, self.ori_action[:,3], label="cmd_real", color=colors[0])
        axsz.plot(self.sim_t, self.action_all.numpy()[:,3], label="cmd_sim", color=colors[3])
        axsz.plot(self.imu_time, self.imu_angle_v[:,2], label="anglex_real", color=colors[2])
        axsz.plot(self.sim_t, self.anglevel.numpy()[:,2], label="anglex_sim",color=colors[1])
        axsz.set_title("anglevel_z")
        # fig2, axs2 = plot.subplots(nrows=1, ncols=1)
        # axs2.plot(self.state_all.numpy()[:,10], label="x_sim", color=colors[0])
        # axs.set_title(f"anglevelx")
        # axs.legend()
        
        # fig_ctbr1, ax = plot.subplots(figsize=(35, 10))
        # ax.plot(self.base_time[2400:2900], self.actions_aligned[2400:2900, 2], label="anglex_cmd_real", color=colors[0])
        # ax.plot(self.base_time[2400:2900], self.anglevel[2400:2900, 1].numpy(), label="anglex_sim", color=colors[1])
        # ax.plot(self.base_time[2400:2900], self.imu_aligned[2400:2900,1], label="anglex_imu_real", color=colors[2])
        # ax.plot(self.base_t, self.actions_aligned[2400:2900, 2], label="anglex_cmd_real", color=colors[0])
        # ax.plot(self.base_time[2400:2900], self.anglevel[2400:2900, 1].numpy(), label="anglex_sim", color=colors[1])
        # ax.plot(self.imu_time, self.imu_aligned[2400:2900,1], label="anglex_imu_real", color=colors[2])
        # ax.set_title("anglevel_y")
        # ax.legend()
        
        # fig_ctbr1, ax = plot.subplots(figsize=(35, 10))
        # ax.plot(self.base_time[2100:2800], self.actions_aligned[2100:2800, 3], label="anglex_cmd_real", color=colors[0])
        # ax.plot(self.base_time[2100:2800], self.anglevel[2100:2800, 2].numpy(), label="anglex_sim", color=colors[1])
        # ax.plot(self.base_time[2100:2800], self.imu_aligned[2100:2800,2], label="anglex_imu_real", color=colors[2])
        # ax.set_title("anglevel_z")
        # ax.legend()
        
        ##############################################################PID_angle_v
        
        # fig2,ax2 = plot.subplots(figsize=(35, 10))
        # ax2.plot(self.imu_angle_v[1400:1600,0], label="anglex_imu_real", color=colors[2])
        # ax2.set_title("anglevel_x")+
        # ax2.legend()
        
        # fig_ctbr2, ay = plot.subplots(figsize=(35, 10))
        # ay.plot(self.actions_time[0:190], self.actions_real[0:190, 2].numpy(), label="angley_real", color=colors[0])
        # # ax.plot(self.actions_time, self.action_drone_aligned[:, 3], label="anglex_drone", color=colors[2])
        # ay.plot(self.actions_time[0:190], self.anglevel[0:190, 1].numpy(), label="angley_sim", color=colors[1])
        # ay.set_title("anglevel_y")
        # ay.legend()
        
        # fig_ctbr3, az = plot.subplots(figsize=(35, 10))
        # az.plot(self.actions_time[0:190], self.actions_real[0:190, 3].numpy(), label="anglez_real", color=colors[0])
        # # ax.plot(self.actions_time, self.action_drone_aligned[:, 3], label="anglex_drone", color=colors[2])
        # az.plot(self.actions_time[0:190], self.anglevel[0:190, 2].numpy(), label="anglez_sim", color=colors[1])
        # az.set_title("anglevel_z")
        # az.legend()
        
        # fig_1_1, posiiton_x = plot.subplots(figsize=(35, 10))
        # posiiton_x.plot(self.actions_time, self.state_all[:,0].numpy(), label=["x_sim"])
        # posiiton_x.plot(self.actions_time, self.state_bf_aligned[:,0], label=["x_real"])
        # posiiton_x.set_title("pos_x")
        # posiiton_x.legend()
        
        # fig_1_2, posiiton_y = plot.subplots(figsize=(35, 10))
        # posiiton_y.plot(self.actions_time, self.state_all[:,1].numpy(), label=["y_sim"])
        # posiiton_y.plot(self.actions_time, self.state_bf_aligned[:,1], label=["y_real"])
        # posiiton_y.set_title("pos_y")
        # posiiton_y.legend()
        
        # fig_1_3, posiiton_z = plot.subplots(figsize=(35, 10))
        # posiiton_z.plot(self.actions_time, self.state_all[:,2].numpy(), label=["z_sim"])
        # posiiton_z.plot(self.actions_time, self.state_bf_aligned[:,2], label=["z_real"])
        # posiiton_z.set_title("pos_z")
        # posiiton_z.legend()
        
        # fig_2, ori_w = plot.subplots(figsize=(35, 10))
        # ori_w.plot(self.state_all[0:200,3].numpy(), label=["qw_sim"])
        # ori_w.plot(self.state_bf_ori_aligned[0:200,0], label=["qw_real"])
        # ori_w.set_title("orientation")
        # ori_w.legend()
        
        # fig_3, ori_x = plot.subplots(figsize=(35, 10))
        # ori_x.plot(self.state_all[0:200,4].numpy(), label=["qx_sim"])
        # ori_x.plot(self.state_bf_ori_aligned[0:200,1], label=["qx_real"])
        # ori_x.set_title("orientation")
        # ori_x.legend()
        
        # fig_4, ori_y = plot.subplots(figsize=(35, 10))
        # ori_y.plot(self.state_all[0:200,5].numpy(), label=["qy_sim"])
        # ori_y.plot(self.state_bf_ori_aligned[0:200,2], label=["qy_real"])
        # ori_y.set_title("orientation")
        # ori_y.legend()
        
        # fig_5, ori_z = plot.subplots(figsize=(35, 10))
        # ori_z.plot(self.state_all[0:200,6].numpy(), label=["qz_sim"])
        # ori_z.plot(self.state_bf_ori_aligned[0:200,3], label=["qz_real"])
        # ori_z.set_title("orientation")
        # ori_z.legend()
        
        plot.show()
        # fig_ctbr.savefig("ctbr.png", dpi=1600)
            
        
bag_data = bag_plot()