import os
import sys

import torch as th

sys.path.append(os.path.dirname(os.getcwd()))

from VisFly.utils.rebot import (  # noqa: E402
    ReBotCLIKParams,
    ReBotIKParams,
    ReBotKinematics,
    TrajPlanParams,
    TrajProfile,
    compute_traj_stats,
    plan_cartesian_geodesic_trajectory,
    plan_joint_space_trajectory,
    solve_pose_ik,
)


def _assert(name, condition):
    if not bool(condition):
        raise AssertionError(name)


def validate_pose_ik():
    kin = ReBotKinematics()
    q_seed = th.tensor([0.0, -0.9, -0.8, 0.4, -0.2, 0.0], dtype=th.float64)
    q_goal = th.tensor([0.25, -0.65, -0.55, 0.35, 0.15, -0.2], dtype=th.float64)
    target = kin.end_effector_transform(q_goal)
    result = solve_pose_ik(
        kin,
        target,
        q_seed,
        ReBotIKParams(max_iter=300, tolerance=1e-5, step_size=0.5, damping=1e-6),
    )
    err = (kin.end_effector_transform(result.q) - target).abs().max()
    _assert("pose IK finite", th.isfinite(result.q).all())
    _assert("pose IK transform error", err < 2e-3)
    print("pose_ik: ok")
    print(f"  success: {result.success}")
    print(f"  iterations: {result.iterations}")
    print(f"  error: {result.error.item():.3e}")
    print(f"  transform max abs: {err.item():.3e}")


def validate_trajectory():
    kin = ReBotKinematics()
    q_start = th.tensor([0.0, -0.9, -0.8, 0.4, -0.2, 0.0], dtype=th.float64)
    q_end = th.tensor([0.2, -0.7, -0.55, 0.25, 0.1, -0.15], dtype=th.float64)
    T_start = kin.end_effector_transform(q_start)
    T_end = kin.end_effector_transform(q_end)
    params = TrajPlanParams(dt=0.05, profile=TrajProfile.MIN_JERK)
    cart = plan_cartesian_geodesic_trajectory(T_start, T_end, duration=1.0, params=params)
    joint_traj = plan_joint_space_trajectory(
        kin,
        q_start,
        q_end,
        duration=1.0,
        params=params,
        ik_params=ReBotCLIKParams(max_iter=80, tolerance=1e-4, damping=1e-6, step_size=0.8),
        null_gain=0.1,
        start_pose=T_start,
        end_pose=T_end,
    )
    stats = compute_traj_stats(kin, joint_traj, T_start, T_end, duration=1.0, params=params)
    _assert("cartesian trajectory point count", cart.n_points == 21)
    _assert("joint trajectory point count", len(joint_traj) == cart.n_points)
    _assert("trajectory finite", th.isfinite(th.stack([pt.q for pt in joint_traj])).all())
    _assert("trajectory average error", stats.avg_ik_error < 2e-2)
    print("trajectory: ok")
    print(f"  points: {len(joint_traj)}")
    print(f"  success_rate: {stats.success_rate:.3f}")
    print(f"  max_error: {stats.max_ik_error:.3e}")
    print(f"  avg_error: {stats.avg_ik_error:.3e}")


def main():
    validate_pose_ik()
    validate_trajectory()


if __name__ == "__main__":
    main()
