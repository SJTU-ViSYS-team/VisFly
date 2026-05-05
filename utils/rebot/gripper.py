from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch as th
import yaml


DEFAULT_REBOT_GRIPPER_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "visfly-beta"
    / "urdf"
    / "rebot_devarm"
    / "gripper.yaml"
)


@dataclass(frozen=True)
class ReBotGripperConfig:
    name: str
    kp: th.Tensor
    kd: th.Tensor
    pos_kp: th.Tensor
    pos_ki: th.Tensor
    vel_kp: th.Tensor
    vel_ki: th.Tensor
    velocity_limit: th.Tensor


@dataclass
class ReBotGripperState:
    q: th.Tensor
    dq: th.Tensor
    tau: th.Tensor


class ReBotGripperDynamics:
    """Differentiable reBot gripper command model aligned with the official API.

    The current open URDF does not expose movable gripper links or joints, so
    this class models the official gripper motor command as a scalar state. It
    can be connected to URDF articulation once the upstream asset provides a
    movable gripper joint.
    """

    def __init__(
        self,
        gripper_config_path: str | Path = DEFAULT_REBOT_GRIPPER_CONFIG,
        dt: float = 0.01,
        mode: str = "pos_vel",
        position_limit: tuple[float, float] = (-th.pi, th.pi),
        max_acceleration: float | th.Tensor = 20.0,
        dtype: th.dtype = th.float64,
        device: th.device | str = "cpu",
    ):
        self.mode = self._canonical_mode(mode)
        self.dt = float(dt)
        self.dtype = dtype
        self.device = th.device(device)
        self.position_limit = th.tensor(position_limit, dtype=dtype, device=self.device)
        self.max_acceleration = th.as_tensor(max_acceleration, dtype=dtype, device=self.device)
        self.config = self._load_config(Path(gripper_config_path))
        self.reset()

    def to(self, device: th.device | str, dtype: Optional[th.dtype] = None) -> "ReBotGripperDynamics":
        self.device = th.device(device)
        self.dtype = self.dtype if dtype is None else dtype
        self.position_limit = self.position_limit.to(device=self.device, dtype=self.dtype)
        self.max_acceleration = self.max_acceleration.to(device=self.device, dtype=self.dtype)
        self.config = ReBotGripperConfig(
            name=self.config.name,
            kp=self.config.kp.to(device=self.device, dtype=self.dtype),
            kd=self.config.kd.to(device=self.device, dtype=self.dtype),
            pos_kp=self.config.pos_kp.to(device=self.device, dtype=self.dtype),
            pos_ki=self.config.pos_ki.to(device=self.device, dtype=self.dtype),
            vel_kp=self.config.vel_kp.to(device=self.device, dtype=self.dtype),
            vel_ki=self.config.vel_ki.to(device=self.device, dtype=self.dtype),
            velocity_limit=self.config.velocity_limit.to(device=self.device, dtype=self.dtype),
        )
        self.state = ReBotGripperState(
            q=self.state.q.to(device=self.device, dtype=self.dtype),
            dq=self.state.dq.to(device=self.device, dtype=self.dtype),
            tau=self.state.tau.to(device=self.device, dtype=self.dtype),
        )
        return self

    def reset(
        self,
        q: Optional[th.Tensor | float] = None,
        dq: Optional[th.Tensor | float] = None,
        tau: Optional[th.Tensor | float] = None,
    ) -> ReBotGripperState:
        q_t = self._scalar(0.0 if q is None else q)
        dq_t = self._scalar(0.0 if dq is None else dq)
        tau_t = self._scalar(0.0 if tau is None else tau)
        self.state = ReBotGripperState(
            q=self._clamp_position(q_t),
            dq=self._clamp_velocity(dq_t),
            tau=tau_t,
        )
        return self.state

    def mode_mit(
        self,
        kp: Optional[th.Tensor | float] = None,
        kd: Optional[th.Tensor | float] = None,
    ) -> bool:
        self.mode = "mit"
        self.config = ReBotGripperConfig(
            name=self.config.name,
            kp=self.config.kp if kp is None else self._scalar(kp),
            kd=self.config.kd if kd is None else self._scalar(kd),
            pos_kp=self.config.pos_kp,
            pos_ki=self.config.pos_ki,
            vel_kp=self.config.vel_kp,
            vel_ki=self.config.vel_ki,
            velocity_limit=self.config.velocity_limit,
        )
        return True

    def mode_pos_vel(self, vlim: Optional[th.Tensor | float] = None) -> bool:
        self.mode = "pos_vel"
        if vlim is not None:
            self.config = ReBotGripperConfig(
                name=self.config.name,
                kp=self.config.kp,
                kd=self.config.kd,
                pos_kp=self.config.pos_kp,
                pos_ki=self.config.pos_ki,
                vel_kp=self.config.vel_kp,
                vel_ki=self.config.vel_ki,
                velocity_limit=self._scalar(vlim),
            )
        return True

    def mode_vel(self) -> bool:
        self.mode = "vel"
        return True

    def mit(
        self,
        pos: th.Tensor | float,
        vel: th.Tensor | float = 0.0,
        kp: Optional[th.Tensor | float] = None,
        kd: Optional[th.Tensor | float] = None,
        tau: th.Tensor | float = 0.0,
        dt: Optional[float] = None,
    ) -> ReBotGripperState:
        old_config = self.config
        if kp is not None or kd is not None:
            self.mode_mit(kp=self.config.kp if kp is None else kp, kd=self.config.kd if kd is None else kd)
        state = self._mit_step(pos=pos, vel=vel, tau=tau, dt=dt)
        self.config = old_config
        return state

    def pos_vel(
        self,
        pos: th.Tensor | float,
        vlim: Optional[th.Tensor | float] = None,
        dt: Optional[float] = None,
    ) -> ReBotGripperState:
        old_config = self.config
        if vlim is not None:
            self.mode_pos_vel(vlim=vlim)
        dt = self.dt if dt is None else float(dt)
        target = self._clamp_position(self._scalar(pos))
        desired_dq = (target - self.state.q) / dt
        dq_next = self._clamp_velocity(desired_dq)
        q_next = self._clamp_position(self.state.q + dq_next * dt)
        self.state = ReBotGripperState(q=q_next, dq=self._clamp_velocity((q_next - self.state.q) / dt), tau=self.state.tau)
        self.config = old_config
        return self.state

    def set_vel(self, vel: th.Tensor | float, dt: Optional[float] = None) -> ReBotGripperState:
        dt = self.dt if dt is None else float(dt)
        dq_next = self._clamp_velocity(self._scalar(vel))
        q_next = self._clamp_position(self.state.q + dq_next * dt)
        self.state = ReBotGripperState(q=q_next, dq=self._clamp_velocity((q_next - self.state.q) / dt), tau=self.state.tau)
        return self.state

    def step(
        self,
        pos: Optional[th.Tensor | float] = None,
        vel: Optional[th.Tensor | float] = None,
        tau: th.Tensor | float = 0.0,
        dt: Optional[float] = None,
    ) -> ReBotGripperState:
        if self.mode == "pos_vel":
            if pos is None:
                pos = self.state.q
            return self.pos_vel(pos, dt=dt)
        if self.mode == "mit":
            if pos is None:
                pos = self.state.q
            if vel is None:
                vel = 0.0
            return self.mit(pos=pos, vel=vel, tau=tau, dt=dt)
        if self.mode == "vel":
            if vel is None:
                vel = 0.0
            return self.set_vel(vel, dt=dt)
        raise RuntimeError(f"Unsupported reBot gripper mode: {self.mode}")

    def _mit_step(
        self,
        pos: th.Tensor | float,
        vel: th.Tensor | float,
        tau: th.Tensor | float,
        dt: Optional[float],
    ) -> ReBotGripperState:
        dt = self.dt if dt is None else float(dt)
        pos_t = self._clamp_position(self._scalar(pos))
        vel_t = self._scalar(vel)
        tau_t = self._scalar(tau)
        ddq = self.config.kp * (pos_t - self.state.q) + self.config.kd * (vel_t - self.state.dq) + tau_t
        ddq = ddq.clamp(-self.max_acceleration, self.max_acceleration)
        dq_next = self._clamp_velocity(self.state.dq + ddq * dt)
        q_next = self._clamp_position(self.state.q + dq_next * dt)
        self.state = ReBotGripperState(q=q_next, dq=self._clamp_velocity((q_next - self.state.q) / dt), tau=tau_t)
        return self.state

    def _canonical_mode(self, mode: str) -> str:
        if mode not in {"mit", "pos_vel", "vel"}:
            raise ValueError("mode should be one of {'mit', 'pos_vel', 'vel'}")
        return mode

    def _clamp_position(self, q: th.Tensor) -> th.Tensor:
        lo, hi = self.position_limit[0].to(q.device, q.dtype), self.position_limit[1].to(q.device, q.dtype)
        return q.clamp(lo, hi)

    def _clamp_velocity(self, dq: th.Tensor) -> th.Tensor:
        limit = self.config.velocity_limit.to(dq.device, dq.dtype)
        return dq.clamp(-limit, limit)

    def _scalar(self, value: th.Tensor | float) -> th.Tensor:
        return th.as_tensor(value, dtype=self.dtype, device=self.device).reshape(())

    def _load_config(self, path: Path) -> ReBotGripperConfig:
        if not path.exists():
            raise FileNotFoundError(f"reBot gripper config not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        gripper = data["gripper"][0]
        mit = gripper["MIT"]
        pos_vel = gripper["POS_VEL"]
        return ReBotGripperConfig(
            name=gripper["name"],
            kp=self._scalar(float(mit["kp"])),
            kd=self._scalar(float(mit["kd"])),
            pos_kp=self._scalar(float(pos_vel["pos_kp"])),
            pos_ki=self._scalar(float(pos_vel["pos_ki"])),
            vel_kp=self._scalar(float(pos_vel["vel_kp"])),
            vel_ki=self._scalar(float(pos_vel["vel_ki"])),
            velocity_limit=self._scalar(float(pos_vel["vlim"])),
        )
