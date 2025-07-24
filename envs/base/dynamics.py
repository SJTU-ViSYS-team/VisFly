import numpy as np
import torch as th
from typing import Union, List, Tuple, Optional, Dict
from abc import ABC
import os, io, sys
import json

sys.path.append(os.getcwd())
from ...utils.maths import Quaternion, Integrator, cross
from ...utils.type import *
import torch.nn as nn
from ...utils.type import ACTION_TYPE

# These will be moved to the correct device in _set_device method
g = th.tensor([[0, 0, -9.81]]).T
z = th.tensor([[0, 0, 1]]).T


class Dynamics:
    action_type_alias: Dict = {"thrust": ACTION_TYPE.THRUST,
                               "bodyrate": ACTION_TYPE.BODYRATE,
                               "velocity": ACTION_TYPE.VELOCITY,
                               "position": ACTION_TYPE.POSITION
                               }

    def __init__(
            self,
            num: int = 1,
            action_type: str = "bodyrate",
            ori_output_type: str = "quaternion",
            seed: int = 42,
            dt: float = 0.005,
            ctrl_dt: float = 0.03,
            ctrl_delay: bool = True,
            comm_delay: float = 0.06,
            action_space: Tuple[float, float] = (-1, 1),
            device: th.device = th.device("cpu"),
            integrator: str = "euler",
            drag_random: float = 0,
            cfg: str = "drone_state",
    ):
        assert action_type in ["bodyrate", "thrust", "velocity", "position"]  # 对两个变量进行断言检查
        assert ori_output_type in ["quaternion", "euler"]

        self.device = device

        # itertive parameters
        self.num = num
        self._position = None
        self._orientation = None
        self._velocity = None
        self._angular_velocity = None
        self._motor_omega = None
        self._thrusts = None
        self._acc = None
        self._angular_acc = None
        self._t = None
        # command format: [gross_thrust / m (z-acc), bodyrate]
        # command format: [yaw, vx, vy, vz]
        # const parameters
        self.action_type = self.action_type_alias[action_type]
        self.angular_output_type = ori_output_type
        # self.angular_dim = 3 if ori_output_type == "" else (4 if ori_output_type == "quaternion" else 6)
        self._is_quat_output = ori_output_type == "quaternion"
        self.dt = dt
        self.ctrl_dt = ctrl_dt
        if not th.as_tensor(ctrl_dt) % th.as_tensor(dt) == 0:
            raise ValueError("ctrl_dt should be a multiple of dt")

        self._interval_steps = int(ctrl_dt / dt)
        self._comm_delay_steps = int(comm_delay / ctrl_dt)
        self._integrator = integrator
        self._ctrl_delay = ctrl_delay

        # initialization
        self.set_seed(seed)
        self._init(cfg=cfg)
        self._get_scale_factor(action_space)
        self._set_device(device)

        self._init_thrust = -(self.m * g / 4)[-1]
        self._init_motor_omega = self._compute_rotor_omega(self._init_thrust)
        
        self._drag_random = drag_random

    def _init(self, cfg):
        self.load(os.path.dirname(__file__) + f"/../../configs/drone/{cfg}.json")
        motor_direction = th.tensor([
            [1, -1, -1, 1, ],
            [-1, -1, 1, 1],
            [0, 0, 0, 0.],
        ])
        motor_direction = motor_direction / motor_direction.norm(dim=0)
        t_BM_ = self._arm_length * motor_direction

        self._inertia = th.diag(self._inertia)

        self._inertia_inv = th.inverse(self._inertia)
        self._B_allocation = th.vstack(
            [th.ones(1, 4), t_BM_[:2], self._kappa * th.tensor([1, -1, 1, -1])]
        )
        self._B_allocation_inv = th.inverse(self._B_allocation)

        self._position = th.zeros((3, self.num), device=self.device)  # 初始位置
        self._orientation = Quaternion(num=self.num, device=self.device)  # 姿态
        self._velocity = th.zeros((3, self.num), device=self.device)  # 速度
        self._angular_velocity = th.zeros((3, self.num), device=self.device)  # 角速度

        self._t = th.zeros((self.num,), device=self.device)

        self._angular_acc = th.zeros((3, self.num), device=self.device)
        # self._ctrl_i = th.zeros((3, self.num), device=self.device)
        # self._pre_action = th.zeros((4, self.num), device=self.device)
        self._pre_action = [
            th.zeros((4, self.num), device=self.device)
            for _ in range(self._comm_delay_steps)
        ]

        self._linear_drag_coeffs = self._linear_drag_coeffs_mean
        self._quad_drag_coeffs = self._quad_drag_coeffs_mean

    def detach(self):
        self._position = self._position.clone().detach()
        self._orientation = self._orientation.clone().detach()
        self._velocity = self._velocity.clone().detach()
        self._angular_velocity = self._angular_velocity.clone().detach()
        self._motor_omega = self._motor_omega.clone().detach()
        self._thrusts = self._thrusts.clone().detach()
        self._angular_acc = self._angular_acc.clone().detach()
        self._acc = self._acc.clone().detach()
        self._t = self._t.clone().detach()
        self._pre_action = [
            pre_act.clone().detach() for pre_act in self._pre_action
        ]

    def _set_device(self, device):
        self._c = self._c.to(device)
        self._thrust_map = self._thrust_map.to(device)
        self._B_allocation = self._B_allocation.to(device)
        self._B_allocation_inv = self._B_allocation_inv.to(device)
        self.m = self.m.to(device)
        self._inertia = self._inertia.to(device)
        self._inertia_inv = self._inertia_inv.to(device)
        self._quad_drag_coeffs_mean = self._quad_drag_coeffs_mean.to(device)
        self._linear_drag_coeffs_mean = self._linear_drag_coeffs_mean.to(device)
        
        # Move drag coefficients if they exist
        if hasattr(self, '_linear_drag_coeffs'):
            self._linear_drag_coeffs = self._linear_drag_coeffs.to(device)
        if hasattr(self, '_quad_drag_coeffs'):
            self._quad_drag_coeffs = self._quad_drag_coeffs.to(device)
        
        # Move PID controllers to the correct device
        self._BODYRATE_PID = self._BODYRATE_PID.to(device)
        self._VELOCITY_PID = self._VELOCITY_PID.to(device)
        self._POSITION_PID = self._POSITION_PID.to(device)

        global z, g
        z = z.to(device)
        g = g.to(device)

    def reset(
            self,
            pos: Union[List, th.Tensor, None] = None,
            ori: Union[List, th.Tensor, None] = None,
            vel: Union[List, th.Tensor, None] = None,
            ori_vel: Union[List, th.Tensor, None] = None,
            motor_omega: Union[List, th.Tensor, None] = None,
            thrusts: Union[List, th.Tensor, None] = None,
            t: Union[List, th.Tensor, None] = None,
            indices: Optional[List] = None,
    ):
        if indices is None:
            self._position = th.zeros((3, self.num), device=self.device) if pos is None else pos.T
            self._orientation = Quaternion(num=self.num, device=self.device) if ori is None else Quaternion(*ori.T)
            self._velocity = th.zeros((3, self.num), device=self.device) if vel is None else vel.T
            self._angular_velocity = th.zeros((3, self.num), device=self.device) if ori_vel is None else ori_vel.T
            self._thrusts = th.ones((4, self.num), device=self.device) * self._init_thrust if thrusts is None else thrusts.T
            self._motor_omega = th.ones((4, self.num), device=self.device) * self._init_motor_omega if motor_omega is None else motor_omega.T
            self._t = th.zeros((self.num,), device=self.device) if t is None else t
            # self._t = th.zeros((self.num,), device=self.device) + th.rand((self.num), device=self.device)*3.14*2 if t is None else t
            # self._ctrl_i = th.zeros((3, self.num), device=self.device)
            self._angular_acc = th.zeros((3, self.num), device=self.device)
            self._acc = th.zeros((3, self.num), device=self.device)
            self._pre_action = [th.zeros(4, self.num) for _ in range(self._comm_delay_steps)]
            if self._drag_random:
                self._linear_drag_coeffs = self._linear_drag_coeffs_mean * (((th.rand_like(self._linear_drag_coeffs_mean)-0.5)*2*self._drag_random).clamp(-0.5, .5) + 1)
                self._quad_drag_coeffs = self._quad_drag_coeffs_mean * (((th.rand_like(self._quad_drag_coeffs_mean)-0.5)*2*self._drag_random).clamp(-0.5, .5) + 1)

        else:
            self._position[:, indices] = th.zeros((3, len(indices)), device=self.device) if pos is None else pos.T
            self._orientation[indices] = Quaternion(num=len(indices), device=self.device) if ori is None else Quaternion(*ori.T)
            self._velocity[:, indices] = th.zeros((3, len(indices)), device=self.device) if vel is None else vel.T
            self._angular_velocity[:, indices] = th.zeros((3, len(indices)), device=self.device) if ori_vel is None else ori_vel.T
            self._motor_omega[:, indices] = th.ones((4, len(indices)), device=self.device) * self._init_motor_omega if motor_omega is None else motor_omega.T
            self._thrusts[:, indices] = th.ones((4, len(indices)), device=self.device) * self._init_thrust if thrusts is None else thrusts.T
            self._t[indices] = th.zeros((len(indices),), device=self.device) if t is None else t
            self._t[indices] = th.zeros((len(indices),), device=self.device) + th.rand((len(indices),)) * 3.14*2 if t is None else t
            # self._ctrl_i[:, indices] = th.zeros((3, len(indices)), device=self.device)
            self._angular_acc[:, indices] = th.zeros((3, len(indices)), device=self.device)
            self._acc[:, indices] = th.zeros((3, len(indices)), device=self.device)
            for i in range(self._comm_delay_steps):
                self._pre_action[i][:, indices] = self._pre_action[i][:, indices] * 0
            
            if self._drag_random:
                self._linear_drag_coeffs[:, indices] = self._linear_drag_coeffs_mean * (((th.rand_like(self._linear_drag_coeffs_mean[:,indices])-0.5)*2*self._drag_random).clamp(-0.5, .5) + 1)
                self._quad_drag_coeffs[:, indices] = self._quad_drag_coeffs_mean * (((th.rand_like(self._quad_drag_coeffs_mean[:,indices])-0.5)*2*self._drag_random).clamp(-0.5, .5) + 1)

        return self.state

    def step(self, action) -> Tuple[th.Tensor, th.Tensor]:

        # Add real imu delay
        if self._comm_delay_steps:
            self._pre_action.append(action.T.clone())
            action = self._pre_action[0].T
            self._pre_action.pop(0)
        else:
            action = action

        command = self._de_normalize(action.to(self.device))

        thrust_des = self._get_thrust_from_cmd(command)  #
        assert (thrust_des <= self._bd_thrust.max).all()  # debug

        for _ in range(self._interval_steps):
            # thrust_des = self._get_thrust_from_cmd(command)
            # assert (thrust_des <= self._bd_thrust.max).all()
            self._run_motors(thrust_des)
            force_torque = self._B_allocation @ self._thrusts  # 计算力矩

            # compute linear acceleration and body torque
            velocity_body = self._orientation.inv_rotate(self._velocity+0)  # (3, N)
            linear_drag = self._linear_drag_coeffs * velocity_body
            quadratic_drag = self._quad_drag_coeffs * velocity_body * velocity_body.abs()
            drag = linear_drag + quadratic_drag
            # drag = self._drag_coeffs * (self._orientation.inv_rotate(self._velocity - 0).pow(2))
            self._acc = self._orientation.rotate(z * force_torque[0] - drag) / self.m + g

            torque = force_torque[1:]

            # integrate the state
            self._position, self._orientation, self._velocity, self._angular_velocity, self._angular_acc = (
                Integrator.integrate(
                    pos=self._position,
                    ori=self._orientation,
                    vel=self._velocity,
                    ori_vel=self._angular_velocity,
                    acc=self._acc,
                    tau=torque,
                    J=self._inertia,
                    J_inv=self._inertia_inv,
                    dt=self.dt,
                    type=self._integrator
                )
            )
            self._orientation = self._orientation.normalize()
        self._t += self.ctrl_dt

        # self._ugly_fix()  # has problems

        return self.state

    def _ugly_fix(self):
        self._position = self._position.clamp(-20, 30)
        self._velocity = self._velocity.clamp(-10, 10)
        self._angular_velocity = self._angular_velocity.clamp(-10, 10)

    def _get_thrust_from_cmd(self, command) -> th.Tensor:
        """_summary_
            get the single _thrusts from the command
        Args:
            command (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.action_type == ACTION_TYPE.THRUST:
            thrusts_des = command
        elif self.action_type == ACTION_TYPE.BODYRATE:
            angular_velocity_error = command[1:] - self._angular_velocity
            # self._ctrl_i += (self._BODYRATE_PID.i @ (angular_velocity_error * self.dt))
            # self._ctrl_i = self._ctrl_i.clip(min=-3, max=3)
            body_torque_des = \
                self._inertia @ self._BODYRATE_PID.p @ angular_velocity_error \
                + cross(self._angular_velocity + 0, self._inertia @ (self._angular_velocity + 0)) \
                - self._BODYRATE_PID.d @ self._angular_acc
            # + self._ctrl_i \

            thrusts_torque = th.cat([command[0:1, :], body_torque_des])
            thrusts_des = self._B_allocation_inv @ thrusts_torque
        elif self.action_type == ACTION_TYPE.VELOCITY:
            command = command.T
            a_des = self._VELOCITY_PID.p * (command[1:] - self._velocity)
            F_des = self.m * (a_des - g)  # world axis

            yaw_spd_des = (self._orientation.toEuler()[2] - command[0]) * self._VELOCITY_PID.p

            gross_thrust_des = self._orientation.transform(F_des)[2]
            R = self._orientation.R
            b3_des = F_des / F_des.norm(dim=0)
            c1_des = th.cat([command[0].cos(), command[0].sin(), th.zeros_like(command[0])])
            b2_des = cross(b3_des, c1_des)
            b2_des = b2_des / b2_des.norm(dim=0)
            b1_des = cross(b2_des, b3_des)
            R_des = th.stack([b1_des, b2_des, b3_des]).transpose(0, 1)

            pose_err = th.zeros_like(self._position)
            ang_vel_err = th.zeros_like(self._position)
            for i in range(self.num):
                m = 0.5 * (R_des[..., i].T @ R[..., i] - R[..., i].T @ R_des[..., i])
                pose_err[:, i] = -th.as_tensor([-m[1, 2], m[0, 2], -m[0, 1]], device=self.device)

                ang_vel_err[:, i] = (R_des[..., i].T @ R[..., i] @ th.tensor([[0], [0], [yaw_spd_des[i]]], device=self.device).squeeze() - self._angular_velocity[:, i])
            body_torque_des = self._inertia @ (self._BODYRATE_PID.p @ pose_err + self._BODYRATE_PID.p @ ang_vel_err - cross(self._angular_velocity, self._angular_velocity))

            thrusts_des = self._B_allocation_inv @ th.vstack([gross_thrust_des, body_torque_des])
            # raise NotImplementedError
        elif self.action_type == ACTION_TYPE.POSITION:
            command = command.T
            v_des = self._POSITION_PID.d * (command[1:] - self._position)
            a_des = self._VELOCITY_PID.d * (v_des - self._velocity)
            F_des = self.m * (a_des - g)  # world axis

            yaw_spd_des = (self._orientation.toEuler()[2] - command[0]) * self._POSITION_PID.d

            gross_thrust_des = self._orientation.transform(F_des)[2]
            R = self._orientation.R
            b3_des = F_des / F_des.norm(dim=0)
            c1_des = th.cat([command[0].cos(), command[0].sin(), th.zeros_like(command[0])])
            b2_des = cross(b3_des, c1_des)
            b2_des = b2_des / b2_des.norm(dim=0)
            b1_des = cross(b2_des, b3_des)
            R_des = th.stack([b1_des, b2_des, b3_des]).transpose(0, 1)

            pose_err = th.zeros_like(self._position)
            ang_vel_err = th.zeros_like(self._position)
            for i in range(self.num):
                m = 0.5 * (R_des[..., i].T @ R[..., i] - R[..., i].T @ R_des[..., i])
                pose_err[:, i] = -th.as_tensor([-m[1, 2], m[0, 2], -m[0, 1]], device=self.device)

                ang_vel_err[:, i] = (R_des[..., i].T @ R[..., i] @ th.tensor([[0], [0], [yaw_spd_des[i]]], device=self.device).squeeze() - self._angular_velocity[:, i])
            body_torque_des = self._inertia @ (self._BODYRATE_PID.p @ pose_err + self._BODYRATE_PID.p @ ang_vel_err - cross(self._angular_velocity, self._angular_velocity))

            thrusts_des = self._B_allocation_inv @ th.vstack([gross_thrust_des, body_torque_des])
            # raise NotImplementedError
        else:
            raise ValueError("action_type should be one of ['thrust', 'bodyrate', 'velocity', 'position']")

        clamp_thrusts_des = th.clamp(thrusts_des, self._bd_thrust.min, self._bd_thrust.max)

        return clamp_thrusts_des

    def _run_motors(self, thrusts_des) -> th.Tensor:
        """_summary_
        Returns:
            _type_: _description_
        """
        if self._ctrl_delay:
            motor_omega_des = self._compute_rotor_omega(thrusts_des)

            # simulate motors as a first-order system
            self._motor_omega = self._c * self._motor_omega + (1 - self._c) * motor_omega_des

            self._thrusts = self._compute_thrust(self._motor_omega)
        else:
            self._thrusts = thrusts_des

        return self._thrusts

    def _compute_thrust(self, _motor_omega) -> th.Tensor:
        """_summary_
            compute the thrust from the motor omega
        Args:
            _motor_omega (_type_): _description_
        Returns:
            _type_: _description_
        """
        _thrusts = (
                (self._thrust_map[0] * (_motor_omega+0).pow(2))
                + self._thrust_map[1] * _motor_omega
                + self._thrust_map[2]
        )
        return _thrusts

    def _compute_rotor_omega(self, _thrusts) -> th.Tensor:
        """_summary_
            compute the rotor omega from the _thrusts by solving quadratic equation
        Args:
            thrusts_des (_type_): _description_
        Returns:
            _type_: _description_
        """
        scale = 1 / (2 * self._thrust_map[0])
        # yi yuan er ci han shu
        omega = scale * (
                -self._thrust_map[1]
                + th.sqrt(
            self._thrust_map[1].pow(2)
            - 4 * self._thrust_map[0] * (self._thrust_map[2] - _thrusts)
        )
        )
        return omega

    def set_seed(self, seed=42):
        th.manual_seed(seed)

    def close(self):
        pass

    def load(self, path=""):
        with open(path, "r") as f:
            data = json.load(f)
        self.m = th.tensor(data["mass"])
        self._cross_sections = th.tensor([data["cross_sections"]]).T  # 机体横截面积
        self._quad_drag_coeffs_mean = th.tensor([data["quad_drag_coeffs"]]).T * 0.5 * 1.225 * self._cross_sections
        self._linear_drag_coeffs_mean = th.tensor([data["linear_drag_coeffs"]]).T
        self._inertia = th.tensor(data["inertia"])
        self.name = data["name"]
        self._BODYRATE_PID = PID(p=th.tensor(data["BODYRAYE_PID"]["p"]),
                                 i=th.tensor(data["BODYRAYE_PID"]["i"]),
                                 d=th.tensor(data["BODYRAYE_PID"]["d"]))
        self._VELOCITY_PID = PID(p=th.tensor(data["VELOCITY_PID"]["p"]), i=th.tensor(data["VELOCITY_PID"]["i"]), d=th.tensor(data["VELOCITY_PID"]["d"]))
        self._POSITION_PID = PID(p=th.tensor(data["POSITION_PID"]["p"]), i=th.tensor(data["POSITION_PID"]["i"]), d=th.tensor(data["POSITION_PID"]["d"]))
        self._kappa = th.tensor(data["kappa"])
        self._arm_length = th.tensor(data["arm_length"])
        self._thrust_map = th.tensor(data["thrust_map"])
        self._motor_tau_inv = th.tensor(1 / data["motor_tau"])
        self._c = th.exp(-self._motor_tau_inv * self.dt)
        self._bd_rotor_omega = bound(
            max=data["motor_omega_max"],
            min=data["motor_omega_min"],
        )
        self._bd_thrust = bound(
            max= \
                self._thrust_map[0] * self._bd_rotor_omega.max ** 2
                + self._thrust_map[1] * self._bd_rotor_omega.max
                + self._thrust_map[2]
            ,
            min=0,
        )
        self._bd_rate = bound(
            max=th.tensor(data["max_rate"]), min=th.tensor(-data["max_rate"])
        )
        self._bd_yaw_rate = bound(
            max=th.tensor(data["max_rate"]), min=th.tensor(-data["max_rate"])
        )
        self._bd_spd = bound(
            max=th.tensor(data["max_spd"]), min=th.tensor(-data["max_spd"])
        )
        self._bd_pos = bound(
            max=th.tensor(data["max_pos"]), min=th.tensor(-data["max_pos"])
        )

    def _get_scale_factor(self, normal_range: Tuple[float, float] = (-1, 1)):
        """_summary_
            get the transformation parameters for the command
        Args:
            normal_range (Tuple[float, float], optional): _description_. Defaults to (-1, 1).
        """
        thrust_normalize_method = "medium"  # "max_min"

        if self.action_type == ACTION_TYPE.BODYRATE:
            if thrust_normalize_method == "medium":
                # (_, average_)
                max_bias = 1
                thrust_scale = (self.m * -g[2]) / self.m
                # thrust_scale = (self.m * -g[2]) * 1 / self.m
                thrust_bias = (self.m * -g[2]) * max_bias / self.m
            elif thrust_normalize_method == "max_min":
                # (min_act, max_act)->(min_thrust, max_thrust) this method try to reach the limit of drone, which is negative for sim2real
                thrust_scale = (
                        (self._bd_thrust.max - self._bd_thrust.min)
                        / self.m
                        / (normal_range[1] - normal_range[0])
                )
                thrust_bias = self._bd_thrust.max / self.m - thrust_scale * normal_range[1]
            else:
                raise ValueError("thrust_normalize_method should be one of ['medium', 'max_min']")

            bodyrate_scale = (self._bd_rate.max - self._bd_rate.min) / (
                    normal_range[1] - normal_range[0]
            )
            bodyrate_bias = self._bd_rate.max - bodyrate_scale * normal_range[1]
            self._normal_params = {
                "thrust": Uniform(mean=thrust_bias, half=thrust_scale).to(self.device),
                "bodyrate": Uniform(mean=bodyrate_bias, half=bodyrate_scale).to(self.device),
            }

        elif self.action_type == ACTION_TYPE.THRUST:
            if thrust_normalize_method == "medium":
                # (_, average_)
                scale = (self.m * -g[2]) / 4 * 2 / self.m
                bias = (self.m * -g[2]) / 4 / self.m
            elif thrust_normalize_method == "max_min":
                scale = (
                        (self._bd_thrust.max - self._bd_thrust.min)
                        / self.m
                        / (normal_range[1] - normal_range[0])
                )
                bias = self._bd_thrust.max / self.m - scale * normal_range[1]
            else:
                raise ValueError("thrust_normalize_method should be one of ['medium', 'max_min']")

            self._normal_params = {"thrust": Uniform(mean=bias, half=scale).to(self.device)}

        elif self.action_type == ACTION_TYPE.VELOCITY:
            spd_scale = (self._bd_spd.max - self._bd_spd.min) / (
                    normal_range[1] - normal_range[0]
            )
            spd_bias = self._bd_spd.max - spd_scale * normal_range[1]
            yaw_scale = th.as_tensor(th.pi - (-th.pi)) / (
                    normal_range[1] - normal_range[0]
            )
            yaw_bias = th.pi - yaw_scale * normal_range[1]
            self._normal_params = {
                "velocity": Uniform(mean=spd_bias, half=spd_scale).to(self.device),
                "yaw": Uniform(mean=yaw_bias, half=yaw_bias).to(self.device),
            }

        elif self.action_type.POSITION:
            pos_scale = (self._bd_pos.max - self._bd_pos.min) / (
                    normal_range[1] - normal_range[0]
            )
            pos_bias = self._bd_pos.max - pos_scale * normal_range[1]
            yaw_scale = th.as_tensor(th.pi - (-th.pi)) / (
                    normal_range[1] - normal_range[0]
            )
            yaw_bias = th.pi - yaw_scale * normal_range[1]
            self._normal_params = {
                "velocity": Uniform(mean=pos_bias, half=pos_scale).to(self.device),
                "yaw": Uniform(mean=yaw_bias, half=yaw_bias).to(self.device),
            }

        else:
            raise ValueError("action_type should be one of ['thrust', 'bodyrate', 'velocity']")

    def _de_normalize(self, command):
        """_summary_
            de-normalize the command to the real value
        Args:
            command (_type_): _description_

        Returns:
            _type_: _description_
        """
        if not isinstance(command, th.Tensor):
            return th.from_numpy(command).T

        if self.action_type == ACTION_TYPE.BODYRATE:
            command = th.hstack([
                (command[:, :1] * self._normal_params["thrust"].half + self._normal_params["thrust"].mean) * self.m,
                command[:, 1:] * self._normal_params["bodyrate"].half + self._normal_params["bodyrate"].mean
            ]
            )
            return command.T

        elif self.action_type == ACTION_TYPE.THRUST:
            command = self.m * (command * self._normal_params["thrust"].half + self._normal_params["thrust"].mean).T
            return command

        elif self.action_type == ACTION_TYPE.VELOCITY:
            command = th.hstack([
                command[:, :1] * self._normal_params["yaw"].half + self._normal_params["yaw"].mean,
                command[:, 1:] * self._normal_params["velocity"].half + self._normal_params["velocity"].mean
            ]
            )
            return command

        elif self.action_type == ACTION_TYPE.POSITION:
            command = th.hstack([
                command[:, :1] * self._normal_params["yaw"].half + self._normal_params["yaw"].mean,
                command[:, 1:] * self._normal_params["velocity"].half + self._normal_params["velocity"].mean
            ]
            )
            return command

        else:
            raise ValueError("action_type should be one of ['thrust', 'bodyrate', 'velocity', 'position']")

    @property
    def position(self):
        return self._position.T

    @property
    def orientation(self):
        if self._is_quat_output:
            return self._orientation.toTensor().T
        else:
            return self._orientation.toEuler().T

    @property
    def direction(self):
        return self._orientation.x_axis.T

    @property
    def velocity(self):
        return self._velocity.T

    @property
    def angular_velocity(self):
        return self._angular_velocity.T

    @property
    def acceleration(self):
        return self._acc.T

    @property
    def angular_acceleration(self):
        return self._angular_acc.T

    @property
    def t(self):
        return self._t

    @property
    def motor_omega(self):
        return self._motor_omega.T

    @property
    def thrusts(self):
        return self._thrusts.T

    @property
    def state(self):
        return th.hstack([
            self.position,
            self.orientation,
            self.velocity,
            self.angular_velocity
        ]
        )

    @property
    def is_quat_output(self):
        return self._is_quat_output

    @property
    def full_state(self):
        return th.hstack([
            self.position,
            self.orientation,
            self.velocity,
            self.angular_velocity,
            self.motor_omega,
            self.thrusts,
            self.t.unsqueeze(1),
        ]
        )

    @property
    def R(self):
        return self._orientation.R

    @property
    def xz_axis(self):
        return self._orientation.xz_axis
