from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch as th

from .kinematics import ReBotKinematics


@dataclass
class ReBotIKParams:
    max_iter: int = 200
    tolerance: float = 1e-4
    step_size: float = 0.5
    damping: float = 1e-6
    backtracking_steps: int = 4


@dataclass
class ReBotIKResult:
    q: th.Tensor
    success: bool
    error: th.Tensor
    iterations: int


def solve_position_ik(
    kinematics: ReBotKinematics,
    target_pos: th.Tensor,
    q_init: th.Tensor,
    params: Optional[ReBotIKParams] = None,
) -> ReBotIKResult:
    """Official-style damped least-squares IK for position-only targets.

    This mirrors the upstream control strategy:
    - damped least squares
    - error-based adaptive damping: lam = damping * max(1, ||err|| * 10)
    - step_size scaling
    - URDF joint-limit clamp
    - backtracking line search accepting only error reduction

    The upstream full pose solver uses Pinocchio log6 and LOCAL frame Jacobian.
    This helper is intentionally position-only for ball tracking, using the
    linear part of the world-aligned geometric Jacobian.
    """
    params = ReBotIKParams() if params is None else params
    q = th.as_tensor(q_init, dtype=kinematics.dtype, device=kinematics.device)
    target_pos = th.as_tensor(target_pos, dtype=q.dtype, device=q.device)

    current_pos = kinematics.end_effector_transform(q)[:3, 3]
    err = target_pos - current_pos
    prev_err = err.norm()

    for iteration in range(params.max_iter):
        if prev_err < params.tolerance:
            return ReBotIKResult(q=q, success=True, error=prev_err, iterations=iteration)

        J = kinematics.geometric_jacobian(q)[:3]
        lam = params.damping * th.maximum(
            th.ones((), dtype=q.dtype, device=q.device),
            prev_err.detach() * 10.0,
        )
        JJT = J @ J.T
        JJT = JJT + lam * th.eye(3, dtype=q.dtype, device=q.device)
        dq = params.step_size * J.T @ th.linalg.solve(JJT, err)

        alpha = 1.0
        accepted = False
        best_q = q
        best_err = prev_err
        best_vec = err
        for _ in range(params.backtracking_steps):
            q_new = kinematics.clamp(q + alpha * dq)
            new_pos = kinematics.end_effector_transform(q_new)[:3, 3]
            err_new = target_pos - new_pos
            new_err = err_new.norm()
            if new_err < prev_err:
                best_q = q_new
                best_err = new_err
                best_vec = err_new
                accepted = True
                break
            alpha *= 0.5

        if accepted:
            q = best_q
            err = best_vec
            prev_err = best_err

    return ReBotIKResult(q=q, success=False, error=prev_err, iterations=params.max_iter)


def solve_pose_ik(
    kinematics: ReBotKinematics,
    target_transform: th.Tensor,
    q_init: th.Tensor,
    params: Optional[ReBotIKParams] = None,
) -> ReBotIKResult:
    """Official-style full SE(3) IK using LOCAL pose error and Jacobian."""
    params = ReBotIKParams() if params is None else params
    q = th.as_tensor(q_init, dtype=kinematics.dtype, device=kinematics.device)
    target_transform = th.as_tensor(target_transform, dtype=q.dtype, device=q.device)

    err = pose_error_local(kinematics.end_effector_transform(q), target_transform)
    prev_err = err.norm()

    for iteration in range(params.max_iter):
        if prev_err < params.tolerance:
            return ReBotIKResult(q=q, success=True, error=prev_err, iterations=iteration)

        J = local_geometric_jacobian(kinematics, q)
        lam = params.damping * th.maximum(
            th.ones((), dtype=q.dtype, device=q.device),
            prev_err.detach() * 10.0,
        )
        JJT = J @ J.T
        JJT = JJT + lam * th.eye(6, dtype=q.dtype, device=q.device)
        dq = params.step_size * J.T @ th.linalg.solve(JJT, err)

        alpha = 1.0
        accepted = False
        best_q = q
        best_err = prev_err
        best_vec = err
        for _ in range(params.backtracking_steps):
            q_new = kinematics.clamp(q + alpha * dq)
            err_new = pose_error_local(kinematics.end_effector_transform(q_new), target_transform)
            new_err = err_new.norm()
            if new_err < prev_err:
                best_q = q_new
                best_err = new_err
                best_vec = err_new
                accepted = True
                break
            alpha *= 0.5

        if accepted:
            q = best_q
            err = best_vec
            prev_err = best_err

    return ReBotIKResult(q=q, success=False, error=prev_err, iterations=params.max_iter)


def make_transform(
    xyz: th.Tensor,
    rpy: Optional[th.Tensor] = None,
    rotation: Optional[th.Tensor] = None,
) -> th.Tensor:
    xyz = th.as_tensor(xyz, dtype=th.float64)
    if rotation is None:
        if rpy is None:
            rotation = th.eye(3, dtype=xyz.dtype, device=xyz.device)
        else:
            rotation = _rpy_to_matrix(th.as_tensor(rpy, dtype=xyz.dtype, device=xyz.device))
    else:
        rotation = th.as_tensor(rotation, dtype=xyz.dtype, device=xyz.device)
    T = th.eye(4, dtype=xyz.dtype, device=xyz.device)
    T[:3, :3] = rotation
    T[:3, 3] = xyz
    return T


def local_geometric_jacobian(kinematics: ReBotKinematics, q: th.Tensor, link_name: str = "end_link") -> th.Tensor:
    T = kinematics.end_effector_transform(q)
    R = T[:3, :3]
    J_world = kinematics.geometric_jacobian(q, link_name)
    J_local = th.zeros_like(J_world)
    J_local[:3] = R.T @ J_world[:3]
    J_local[3:] = R.T @ J_world[3:]
    return J_local


def pose_error_local(current_transform: th.Tensor, target_transform: th.Tensor) -> th.Tensor:
    T_err = _inverse_transform(current_transform) @ target_transform
    return log_se3(T_err)


def log_se3(T: th.Tensor) -> th.Tensor:
    R = T[:3, :3]
    t = T[:3, 3]
    omega = log_so3(R)
    theta = omega.norm()
    wx = _skew(omega)
    eye = th.eye(3, dtype=T.dtype, device=T.device)
    if float(theta.detach()) < 1e-8:
        v = (eye - 0.5 * wx + (wx @ wx) / 12.0) @ t
    else:
        half_theta = 0.5 * theta
        cot_half = th.cos(half_theta) / th.sin(half_theta)
        v_inv = eye - 0.5 * wx + (1.0 - theta * cot_half / 2.0) / (theta * theta) * (wx @ wx)
        v = v_inv @ t
    return th.cat([v, omega], dim=0)


def exp_se3(xi: th.Tensor) -> th.Tensor:
    v = xi[:3]
    omega = xi[3:]
    theta = omega.norm()
    wx = _skew(omega)
    eye = th.eye(3, dtype=xi.dtype, device=xi.device)
    R = exp_so3(omega)
    if float(theta.detach()) < 1e-8:
        V = eye + 0.5 * wx + (wx @ wx) / 6.0
    else:
        V = eye + (1.0 - th.cos(theta)) / (theta * theta) * wx
        V = V + (theta - th.sin(theta)) / (theta * theta * theta) * (wx @ wx)
    T = th.eye(4, dtype=xi.dtype, device=xi.device)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def log_so3(R: th.Tensor) -> th.Tensor:
    cos_theta = ((th.trace(R) - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = th.acos(cos_theta)
    vee = th.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if float(theta.detach()) < 1e-8:
        return 0.5 * vee
    return theta / (2.0 * th.sin(theta).clamp_min(1e-12)) * vee


def exp_so3(omega: th.Tensor) -> th.Tensor:
    theta = omega.norm()
    wx = _skew(omega)
    eye = th.eye(3, dtype=omega.dtype, device=omega.device)
    if float(theta.detach()) < 1e-8:
        return eye + wx + 0.5 * (wx @ wx)
    return eye + th.sin(theta) / theta * wx + (1.0 - th.cos(theta)) / (theta * theta) * (wx @ wx)


def _inverse_transform(T: th.Tensor) -> th.Tensor:
    inv = th.eye(4, dtype=T.dtype, device=T.device)
    R = T[:3, :3]
    p = T[:3, 3]
    inv[:3, :3] = R.T
    inv[:3, 3] = -(R.T @ p)
    return inv


def _skew(v: th.Tensor) -> th.Tensor:
    return th.stack(
        [
            th.stack([th.zeros_like(v[0]), -v[2], v[1]]),
            th.stack([v[2], th.zeros_like(v[0]), -v[0]]),
            th.stack([-v[1], v[0], th.zeros_like(v[0])]),
        ]
    )


def _rpy_to_matrix(rpy: th.Tensor) -> th.Tensor:
    roll, pitch, yaw = rpy
    cr, sr = th.cos(roll), th.sin(roll)
    cp, sp = th.cos(pitch), th.sin(pitch)
    cy, sy = th.cos(yaw), th.sin(yaw)
    rx = th.stack(
        [
            th.stack([th.ones_like(roll), th.zeros_like(roll), th.zeros_like(roll)]),
            th.stack([th.zeros_like(roll), cr, -sr]),
            th.stack([th.zeros_like(roll), sr, cr]),
        ]
    )
    ry = th.stack(
        [
            th.stack([cp, th.zeros_like(roll), sp]),
            th.stack([th.zeros_like(roll), th.ones_like(roll), th.zeros_like(roll)]),
            th.stack([-sp, th.zeros_like(roll), cp]),
        ]
    )
    rz = th.stack(
        [
            th.stack([cy, -sy, th.zeros_like(roll)]),
            th.stack([sy, cy, th.zeros_like(roll)]),
            th.stack([th.zeros_like(roll), th.zeros_like(roll), th.ones_like(roll)]),
        ]
    )
    return rz @ ry @ rx
