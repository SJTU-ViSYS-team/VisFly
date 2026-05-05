import os
import sys

import numpy as np
import torch as th

sys.path.append(os.path.dirname(os.getcwd()))

from VisFly.utils.rebot import (  # noqa: E402
    ReBotKinematics,
    TrajPlanParams,
    TrajProfile,
    plan_cartesian_geodesic_trajectory,
)
from VisFly.utils.rebot.ik import exp_se3, log_se3, pose_error_local  # noqa: E402


def _load_pinocchio():
    try:
        import pinocchio as pin
    except ImportError as exc:
        raise SystemExit("Pinocchio is not installed. Install `pin` to run this comparison.") from exc
    return pin


def _to_pin_se3(pin, T: th.Tensor):
    T_np = T.detach().cpu().numpy()
    return pin.SE3(T_np[:3, :3], T_np[:3, 3])


def _from_pin_se3(T):
    matrix = np.eye(4)
    matrix[:3, :3] = T.rotation
    matrix[:3, 3] = T.translation
    return matrix


def main():
    pin = _load_pinocchio()
    kin = ReBotKinematics()
    q_start = th.tensor([0.0, -0.9, -0.8, 0.4, -0.2, 0.0], dtype=th.float64)
    q_end = th.tensor([0.2, -0.7, -0.55, 0.25, 0.1, -0.15], dtype=th.float64)
    T_start = kin.end_effector_transform(q_start)
    T_end = kin.end_effector_transform(q_end)

    start_pin = _to_pin_se3(pin, T_start)
    end_pin = _to_pin_se3(pin, T_end)
    delta_pin = pin.log6(start_pin.inverse() * end_pin).vector
    delta_model = log_se3(th.linalg.inv(T_start) @ T_end).detach().numpy()
    log_error = float(np.linalg.norm(delta_model - delta_pin, ord=np.inf))

    exp_model = exp_se3(th.tensor(delta_pin, dtype=th.float64)).detach().numpy()
    exp_pin = _from_pin_se3(pin.exp6(delta_pin))
    exp_error = float(np.linalg.norm(exp_model - exp_pin, ord=np.inf))

    params = TrajPlanParams(dt=0.05, profile=TrajProfile.MIN_JERK)
    traj = plan_cartesian_geodesic_trajectory(T_start, T_end, 1.0, params).trajectory.points()
    max_interp_error = 0.0
    for pt in traj:
        s_pose = _to_pin_se3(pin, pt.pose)
        s = pin.log6(start_pin.inverse() * s_pose).vector
        if np.linalg.norm(delta_pin) > 1e-12:
            alpha = float(np.dot(s, delta_pin) / np.dot(delta_pin, delta_pin))
        else:
            alpha = 0.0
        ref = start_pin * pin.exp6(delta_pin * alpha)
        max_interp_error = max(max_interp_error, float(np.linalg.norm(pt.pose.numpy() - _from_pin_se3(ref), ord=np.inf)))

    pose_error = pose_error_local(T_start, T_end).detach().numpy()
    pose_error_diff = float(np.linalg.norm(pose_error - delta_pin, ord=np.inf))

    print(f"log_error_inf: {log_error:.3e}")
    print(f"exp_error_inf: {exp_error:.3e}")
    print(f"pose_error_inf: {pose_error_diff:.3e}")
    print(f"interp_error_inf: {max_interp_error:.3e}")
    if max(log_error, exp_error, pose_error_diff, max_interp_error) > 1e-8:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
