from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch as th
import yaml

from .kinematics import ReBotKinematics


DEFAULT_REBOT_ARM_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "visfly-beta"
    / "urdf"
    / "rebot_devarm"
    / "arm.yaml"
)


@dataclass(frozen=True)
class ReBotMotorConfig:
    joint_names: list[str]
    kp: th.Tensor
    kd: th.Tensor
    pos_kp: th.Tensor
    pos_ki: th.Tensor
    vel_kp: th.Tensor
    vel_ki: th.Tensor
    velocity_limit: th.Tensor


@dataclass
class ReBotJointState:
    q: th.Tensor
    dq: th.Tensor


class ReBotJointDynamics:
    """Differentiable joint-space dynamics approximation for reBot.

    This class intentionally stays in joint space. It does not model contact or
    full rigid-body dynamics. Its purpose is to keep commanded arm motion inside
    URDF joint limits and motor-configured velocity/acceleration envelopes while
    remaining PyTorch differentiable.
    """

    def __init__(
        self,
        kinematics: Optional[ReBotKinematics] = None,
        arm_config_path: str | Path = DEFAULT_REBOT_ARM_CONFIG,
        dt: float = 0.03,
        mode: str = "pos_vel",
        max_acceleration: float | th.Tensor = 20.0,
        dtype: th.dtype = th.float64,
        device: th.device | str = "cpu",
    ):
        mode = self._canonical_mode(mode)

        self.kinematics = kinematics if kinematics is not None else ReBotKinematics(dtype=dtype, device=device)
        self.dt = float(dt)
        self.mode = mode
        self.dtype = dtype
        self.device = th.device(device)
        self.motor_config = self._load_motor_config(Path(arm_config_path))
        self.max_acceleration = th.as_tensor(max_acceleration, dtype=dtype, device=self.device)
        if self.max_acceleration.ndim == 0:
            self.max_acceleration = self.max_acceleration.repeat(6)
        self.reset()

    def to(self, device: th.device | str, dtype: Optional[th.dtype] = None) -> "ReBotJointDynamics":
        self.device = th.device(device)
        self.dtype = self.dtype if dtype is None else dtype
        self.kinematics.to(device=self.device, dtype=self.dtype)
        self.motor_config = ReBotMotorConfig(
            joint_names=self.motor_config.joint_names,
            kp=self.motor_config.kp.to(device=self.device, dtype=self.dtype),
            kd=self.motor_config.kd.to(device=self.device, dtype=self.dtype),
            pos_kp=self.motor_config.pos_kp.to(device=self.device, dtype=self.dtype),
            pos_ki=self.motor_config.pos_ki.to(device=self.device, dtype=self.dtype),
            vel_kp=self.motor_config.vel_kp.to(device=self.device, dtype=self.dtype),
            vel_ki=self.motor_config.vel_ki.to(device=self.device, dtype=self.dtype),
            velocity_limit=self.motor_config.velocity_limit.to(device=self.device, dtype=self.dtype),
        )
        self.max_acceleration = self.max_acceleration.to(device=self.device, dtype=self.dtype)
        self.state = ReBotJointState(
            q=self.state.q.to(device=self.device, dtype=self.dtype),
            dq=self.state.dq.to(device=self.device, dtype=self.dtype),
        )
        return self

    def reset(
        self,
        q: Optional[th.Tensor] = None,
        dq: Optional[th.Tensor] = None,
    ) -> ReBotJointState:
        if q is None:
            q = th.zeros(6, dtype=self.dtype, device=self.device)
        else:
            q = th.as_tensor(q, dtype=self.dtype, device=self.device)
        if dq is None:
            dq = th.zeros_like(q)
        else:
            dq = th.as_tensor(dq, dtype=self.dtype, device=self.device)
        self.state = ReBotJointState(q=self.kinematics.clamp(q), dq=self._clamp_velocity(dq))
        return self.state

    def step(
        self,
        q_target: th.Tensor,
        dq_target: Optional[th.Tensor] = None,
        dt: Optional[float] = None,
    ) -> ReBotJointState:
        dt = self.dt if dt is None else float(dt)
        q_target = self.kinematics.clamp(th.as_tensor(q_target, dtype=self.dtype, device=self.device))
        if dq_target is None:
            dq_target = th.zeros_like(q_target)
        else:
            dq_target = th.as_tensor(dq_target, dtype=self.dtype, device=self.device)

        q = self.state.q
        dq = self.state.dq
        if self.mode == "pos_vel":
            dq_next = self._velocity_to_target(q, q_target, dt)
        elif self.mode == "mit":
            kp = self.motor_config.kp.to(q.device, q.dtype)
            kd = self.motor_config.kd.to(q.device, q.dtype)
            ddq = kp * (q_target - q) + kd * (dq_target - dq)
            ddq = self._clamp_acceleration(ddq)
            dq_next = self._clamp_velocity(dq + ddq * dt)
        elif self.mode == "vel":
            dq_next = self._clamp_velocity(dq_target)
        else:
            raise RuntimeError(f"Unsupported reBot dynamics mode: {self.mode}")

        q_next = self.kinematics.clamp(q + dq_next * dt)
        dq_next = self._clamp_velocity((q_next - q) / dt)
        self.state = ReBotJointState(q=q_next, dq=dq_next)
        return self.state

    def mode_mit(
        self,
        kp: Optional[th.Tensor] = None,
        kd: Optional[th.Tensor] = None,
    ) -> bool:
        """Switch to the official MIT motor command mode."""
        self.mode = "mit"
        if kp is not None:
            self.motor_config = ReBotMotorConfig(
                joint_names=self.motor_config.joint_names,
                kp=th.as_tensor(kp, dtype=self.dtype, device=self.device),
                kd=self.motor_config.kd,
                pos_kp=self.motor_config.pos_kp,
                pos_ki=self.motor_config.pos_ki,
                vel_kp=self.motor_config.vel_kp,
                vel_ki=self.motor_config.vel_ki,
                velocity_limit=self.motor_config.velocity_limit,
            )
        if kd is not None:
            self.motor_config = ReBotMotorConfig(
                joint_names=self.motor_config.joint_names,
                kp=self.motor_config.kp,
                kd=th.as_tensor(kd, dtype=self.dtype, device=self.device),
                pos_kp=self.motor_config.pos_kp,
                pos_ki=self.motor_config.pos_ki,
                vel_kp=self.motor_config.vel_kp,
                vel_ki=self.motor_config.vel_ki,
                velocity_limit=self.motor_config.velocity_limit,
            )
        return True

    def mode_pos_vel(self, vlim: Optional[th.Tensor] = None) -> bool:
        """Switch to the official POS_VEL motor command mode."""
        self.mode = "pos_vel"
        if vlim is not None:
            self.motor_config = ReBotMotorConfig(
                joint_names=self.motor_config.joint_names,
                kp=self.motor_config.kp,
                kd=self.motor_config.kd,
                pos_kp=self.motor_config.pos_kp,
                pos_ki=self.motor_config.pos_ki,
                vel_kp=self.motor_config.vel_kp,
                vel_ki=self.motor_config.vel_ki,
                velocity_limit=th.as_tensor(vlim, dtype=self.dtype, device=self.device),
            )
        return True

    def mode_vel(self) -> bool:
        """Switch to the official pure velocity command mode."""
        self.mode = "vel"
        return True

    def mit(
        self,
        pos: th.Tensor,
        vel: Optional[th.Tensor] = None,
        kp: Optional[th.Tensor] = None,
        kd: Optional[th.Tensor] = None,
        tau: Optional[th.Tensor] = None,
        dt: Optional[float] = None,
    ) -> ReBotJointState:
        """Official-style MIT command.

        The hardware command accepts a torque feedforward term. In this
        joint-space differentiable approximation, tau is treated as an
        acceleration feedforward term with the same per-joint command shape.
        Use ReBotRigidBodyDynamics for torque-level rigid-body simulation.
        """
        previous_mode = self.mode
        self.mode = "mit"
        if vel is None:
            vel = th.zeros_like(th.as_tensor(pos, dtype=self.dtype, device=self.device))
        if tau is None:
            tau = th.zeros_like(th.as_tensor(pos, dtype=self.dtype, device=self.device))
        old_config = self.motor_config
        if kp is not None or kd is not None:
            self.mode_mit(
                self.motor_config.kp if kp is None else th.as_tensor(kp, dtype=self.dtype, device=self.device),
                self.motor_config.kd if kd is None else th.as_tensor(kd, dtype=self.dtype, device=self.device),
            )
        state = self._mit_step(pos, vel, tau, dt)
        self.motor_config = old_config
        self.mode = previous_mode
        return state

    def pos_vel(
        self,
        pos: th.Tensor,
        vlim: Optional[th.Tensor] = None,
        dt: Optional[float] = None,
    ) -> ReBotJointState:
        """Official-style POS_VEL command."""
        previous_mode = self.mode
        old_config = self.motor_config
        self.mode_pos_vel(vlim)
        state = self.step(pos, dt=dt)
        self.motor_config = old_config
        self.mode = previous_mode
        return state

    def set_vel(self, vel: th.Tensor, dt: Optional[float] = None) -> ReBotJointState:
        """Official-style pure velocity command."""
        previous_mode = self.mode
        self.mode = "vel"
        state = self.step(self.state.q, dq_target=vel, dt=dt)
        self.mode = previous_mode
        return state

    def gravity_compensation(
        self,
        tau_g: th.Tensor,
        kd: float | th.Tensor = 1.0,
        dt: Optional[float] = None,
    ) -> ReBotJointState:
        """Official-style gravity compensation command.

        Mirrors the upstream example:
            q = current joint position
            tau = g(q)
            pos = q
            vel = 0
            kp = 0
            kd = 1
        """
        q = self.state.q
        return self.mit(
            pos=q,
            vel=th.zeros_like(q),
            kp=th.zeros_like(q),
            kd=th.as_tensor(kd, dtype=q.dtype, device=q.device).expand_as(q),
            tau=tau_g,
            dt=dt,
        )

    def rollout(
        self,
        q_targets: th.Tensor,
        dq_targets: Optional[th.Tensor] = None,
        q0: Optional[th.Tensor] = None,
        dq0: Optional[th.Tensor] = None,
    ) -> ReBotJointState:
        self.reset(q0, dq0)
        qs = []
        dqs = []
        if dq_targets is None:
            dq_targets = [None] * len(q_targets)
        for q_target, dq_target in zip(q_targets, dq_targets):
            state = self.step(q_target, dq_target)
            qs.append(state.q)
            dqs.append(state.dq)
        return ReBotJointState(q=th.stack(qs), dq=th.stack(dqs))

    def end_effector_transform(self) -> th.Tensor:
        return self.kinematics.end_effector_transform(self.state.q)

    def _velocity_to_target(self, q: th.Tensor, q_target: th.Tensor, dt: float) -> th.Tensor:
        desired_dq = (q_target - q) / dt
        return self._clamp_velocity(desired_dq)

    def _mit_step(
        self,
        pos: th.Tensor,
        vel: th.Tensor,
        tau: th.Tensor,
        dt: Optional[float],
    ) -> ReBotJointState:
        dt = self.dt if dt is None else float(dt)
        pos = self.kinematics.clamp(th.as_tensor(pos, dtype=self.dtype, device=self.device))
        vel = th.as_tensor(vel, dtype=self.dtype, device=self.device)
        tau = th.as_tensor(tau, dtype=self.dtype, device=self.device)
        q = self.state.q
        dq = self.state.dq
        kp = self.motor_config.kp.to(q.device, q.dtype)
        kd = self.motor_config.kd.to(q.device, q.dtype)
        ddq = kp * (pos - q) + kd * (vel - dq) + tau
        ddq = self._clamp_acceleration(ddq)
        dq_next = self._clamp_velocity(dq + ddq * dt)
        q_next = self.kinematics.clamp(q + dq_next * dt)
        dq_next = self._clamp_velocity((q_next - q) / dt)
        self.state = ReBotJointState(q=q_next, dq=dq_next)
        return self.state

    def _canonical_mode(self, mode: str) -> str:
        if mode not in {"mit", "pos_vel", "vel"}:
            raise ValueError("mode should be one of {'mit', 'pos_vel', 'vel'}")
        return mode

    def _clamp_velocity(self, dq: th.Tensor) -> th.Tensor:
        limit = self.motor_config.velocity_limit.to(dq.device, dq.dtype)
        return dq.clamp(-limit, limit)

    def _clamp_acceleration(self, ddq: th.Tensor) -> th.Tensor:
        limit = self.max_acceleration.to(ddq.device, ddq.dtype)
        return ddq.clamp(-limit, limit)

    def _load_motor_config(self, path: Path) -> ReBotMotorConfig:
        if not path.exists():
            raise FileNotFoundError(f"reBot arm motor config not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        by_name: Dict[str, dict] = {joint["name"]: joint for joint in data["joints"]}
        kp = []
        kd = []
        pos_kp = []
        pos_ki = []
        vel_kp = []
        vel_ki = []
        velocity_limit = []
        resolved_names = []
        for urdf_joint_name in self.kinematics.joint_names:
            config_name = "joint3" if urdf_joint_name == "join3" else urdf_joint_name
            if config_name not in by_name:
                raise KeyError(f"Missing motor config for URDF joint {urdf_joint_name}")
            joint = by_name[config_name]
            resolved_names.append(config_name)
            kp.append(float(joint["MIT"]["kp"]))
            kd.append(float(joint["MIT"]["kd"]))
            pos_kp.append(float(joint["POS_VEL"]["pos_kp"]))
            pos_ki.append(float(joint["POS_VEL"]["pos_ki"]))
            vel_kp.append(float(joint["POS_VEL"]["vel_kp"]))
            vel_ki.append(float(joint["POS_VEL"]["vel_ki"]))
            velocity_limit.append(float(joint["POS_VEL"]["vlim"]))

        return ReBotMotorConfig(
            joint_names=resolved_names,
            kp=th.tensor(kp, dtype=self.dtype, device=self.device),
            kd=th.tensor(kd, dtype=self.dtype, device=self.device),
            pos_kp=th.tensor(pos_kp, dtype=self.dtype, device=self.device),
            pos_ki=th.tensor(pos_ki, dtype=self.dtype, device=self.device),
            vel_kp=th.tensor(vel_kp, dtype=self.dtype, device=self.device),
            vel_ki=th.tensor(vel_ki, dtype=self.dtype, device=self.device),
            velocity_limit=th.tensor(velocity_limit, dtype=self.dtype, device=self.device),
        )
