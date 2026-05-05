from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional

import torch as th

from .ik import exp_se3, local_geometric_jacobian, log_se3, pose_error_local
from .kinematics import ReBotKinematics


class TrajProfile(enum.Enum):
    LINEAR = "linear"
    MIN_JERK = "min_jerk"
    TRAPEZOID = "trapezoid"


@dataclass
class TrajPlanParams:
    dt: float = 0.02
    profile: TrajProfile = TrajProfile.MIN_JERK
    accel_ratio: float = 0.25


@dataclass
class ReBotCLIKParams:
    max_iter: int = 200
    tolerance: float = 1e-4
    damping: float = 1e-6
    step_size: float = 0.8


@dataclass
class CartesianPoint:
    time: float
    pose: th.Tensor


@dataclass
class CartesianTrajectory:
    points_: List[CartesianPoint] = field(default_factory=list)

    def add_point(self, t: float, pose: th.Tensor) -> None:
        self.points_.append(CartesianPoint(t, pose))

    def duration(self) -> float:
        return self.points_[-1].time if self.points_ else 0.0

    def points(self) -> List[CartesianPoint]:
        return self.points_


@dataclass
class CartesianTrajectoryResult:
    trajectory: CartesianTrajectory
    n_points: int


@dataclass
class JointTrajectoryPoint:
    time: float
    q: th.Tensor
    ik_success: bool


@dataclass
class TrajStats:
    total_points: int = 0
    success_count: int = 0
    success_rate: float = 0.0
    max_ik_error: float = 0.0
    avg_ik_error: float = 0.0


def plan_cartesian_geodesic_trajectory(
    start_pose: th.Tensor,
    end_pose: th.Tensor,
    duration: float,
    params: Optional[TrajPlanParams] = None,
) -> CartesianTrajectoryResult:
    if duration <= 0.0:
        raise ValueError("duration must be > 0")
    params = TrajPlanParams() if params is None else params
    start_pose = th.as_tensor(start_pose, dtype=th.float64)
    end_pose = th.as_tensor(end_pose, dtype=start_pose.dtype, device=start_pose.device)

    traj = CartesianTrajectory()
    n = max(2, int(th.ceil(th.tensor(duration / params.dt)).item()) + 1)
    dt = duration / (n - 1)
    for i in range(n):
        t = i * dt
        s = _apply_profile(t / duration, params.profile, params.accel_ratio)
        traj.add_point(t, se3_interpolate(start_pose, end_pose, s))
    return CartesianTrajectoryResult(trajectory=traj, n_points=n)


def track_trajectory(
    kinematics: ReBotKinematics,
    trajectory: CartesianTrajectory,
    q_init: th.Tensor,
    ik_params: Optional[ReBotCLIKParams] = None,
    null_gain: float = 0.0,
) -> List[JointTrajectoryPoint]:
    ik_params = ReBotCLIKParams() if ik_params is None else ik_params
    q = th.as_tensor(q_init, dtype=kinematics.dtype, device=kinematics.device)
    result: List[JointTrajectoryPoint] = []

    for pt in trajectory.points():
        converged = False
        target = pt.pose.to(device=q.device, dtype=q.dtype)
        for _ in range(ik_params.max_iter):
            err = pose_error_local(kinematics.end_effector_transform(q), target)
            err_norm = err.norm()
            if err_norm < ik_params.tolerance:
                converged = True
                break

            J = local_geometric_jacobian(kinematics, q)
            lam = ik_params.damping * th.maximum(
                th.ones((), dtype=q.dtype, device=q.device),
                err_norm.detach() * 10.0,
            )
            JJT = J @ J.T
            JJT = JJT + lam * th.eye(6, dtype=q.dtype, device=q.device)
            dq = ik_params.step_size * J.T @ th.linalg.solve(JJT, err)

            if null_gain > 0.0:
                g = joint_limit_gradient(kinematics, q)
                dq = dq + null_gain * (g - J.T @ th.linalg.solve(JJT, J @ g))

            q = kinematics.clamp(q + dq)

        result.append(JointTrajectoryPoint(pt.time, q.clone(), converged))

    return result


def plan_joint_space_trajectory(
    kinematics: ReBotKinematics,
    q_start: th.Tensor,
    q_end: th.Tensor,
    duration: float,
    params: Optional[TrajPlanParams] = None,
    ik_params: Optional[ReBotCLIKParams] = None,
    null_gain: float = 0.0,
    start_pose: Optional[th.Tensor] = None,
    end_pose: Optional[th.Tensor] = None,
) -> List[JointTrajectoryPoint]:
    if duration <= 0.0:
        raise ValueError("duration must be > 0")
    q_start = th.as_tensor(q_start, dtype=kinematics.dtype, device=kinematics.device)
    q_end = th.as_tensor(q_end, dtype=kinematics.dtype, device=kinematics.device)
    T_start = kinematics.end_effector_transform(q_start) if start_pose is None else start_pose
    T_end = kinematics.end_effector_transform(q_end) if end_pose is None else end_pose
    cart_result = plan_cartesian_geodesic_trajectory(T_start, T_end, duration, params)
    return track_trajectory(kinematics, cart_result.trajectory, q_start, ik_params, null_gain)


def compute_traj_stats(
    kinematics: ReBotKinematics,
    joint_trajectory: List[JointTrajectoryPoint],
    start_pose: th.Tensor,
    end_pose: th.Tensor,
    duration: float,
    params: Optional[TrajPlanParams] = None,
) -> TrajStats:
    stats = TrajStats(total_points=len(joint_trajectory))
    ref_result = plan_cartesian_geodesic_trajectory(start_pose, end_pose, duration, params)
    ref_pts = ref_result.trajectory.points()
    sum_err = 0.0
    for i, pt in enumerate(joint_trajectory):
        if i >= len(ref_pts):
            break
        if pt.ik_success:
            stats.success_count += 1
        err_norm = pose_error_local(kinematics.end_effector_transform(pt.q), ref_pts[i].pose).norm()
        err_float = float(err_norm.detach())
        stats.max_ik_error = max(stats.max_ik_error, err_float)
        sum_err += err_float
    if stats.total_points > 0:
        stats.success_rate = stats.success_count / stats.total_points
        stats.avg_ik_error = sum_err / stats.total_points
    return stats


def se3_interpolate(start_pose: th.Tensor, end_pose: th.Tensor, s: float) -> th.Tensor:
    delta = log_se3(_inverse_transform(start_pose) @ end_pose)
    return start_pose @ exp_se3(delta * float(s))


def joint_limit_gradient(kinematics: ReBotKinematics, q: th.Tensor) -> th.Tensor:
    lo, hi = kinematics.joint_limits
    lo = lo.to(q.device, q.dtype)
    hi = hi.to(q.device, q.dtype)
    dl = q - lo
    dh = hi - q
    mask = (dl > 1e-6) & (dh > 1e-6)
    g = th.zeros_like(q)
    g = th.where(mask, (dh - dl) / (dl * dh), g)
    return g


def _apply_profile(t: float, profile: TrajProfile, accel_ratio: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    if profile == TrajProfile.LINEAR:
        return t
    if profile == TrajProfile.MIN_JERK:
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        return 10.0 * t3 - 15.0 * t4 + 6.0 * t5
    if profile == TrajProfile.TRAPEZOID:
        ta = max(0.01, min(0.49, accel_ratio))
        vm = 2.0 / (1.0 - ta)
        if t <= ta:
            return 0.5 * vm / ta * t * t
        if t <= 1.0 - ta:
            return 0.5 * vm * ta + vm * (t - ta)
        dt = 1.0 - t
        return 1.0 - 0.5 * vm / ta * dt * dt
    return t


def _inverse_transform(T: th.Tensor) -> th.Tensor:
    inv = th.eye(4, dtype=T.dtype, device=T.device)
    R = T[:3, :3]
    p = T[:3, 3]
    inv[:3, :3] = R.T
    inv[:3, 3] = -(R.T @ p)
    return inv
