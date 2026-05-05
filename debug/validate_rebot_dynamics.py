import os
import sys

import torch as th

sys.path.append(os.path.dirname(os.getcwd()))

from VisFly.utils.rebot import ReBotJointDynamics, ReBotKinematics  # noqa: E402


def _assert(name, condition):
    if not bool(condition):
        raise AssertionError(name)


def validate_mode(mode: str):
    kin = ReBotKinematics()
    dynamics = ReBotJointDynamics(kinematics=kin, mode=mode, dt=0.03)
    lower, upper = kin.joint_limits
    vlim = dynamics.motor_config.velocity_limit

    q0 = th.zeros(6, dtype=th.float64)
    dynamics.reset(q0)
    target = upper + 10.0
    states = []
    for _ in range(100):
        if mode == "pos_vel":
            states.append(dynamics.pos_vel(target))
        elif mode == "mit":
            states.append(dynamics.mit(target))
        elif mode == "vel":
            states.append(dynamics.set_vel(th.ones(6, dtype=th.float64)))
        else:
            raise RuntimeError(mode)

    q = dynamics.state.q
    dq = dynamics.state.dq
    _assert(f"{mode}: q lower limit", (q >= lower - 1e-12).all())
    _assert(f"{mode}: q upper limit", (q <= upper + 1e-12).all())
    _assert(f"{mode}: dq velocity limit", (dq.abs() <= vlim + 1e-12).all())

    q_targets = th.stack([target * (i + 1) / 20.0 for i in range(20)]).requires_grad_(True)
    if mode == "vel":
        dynamics.reset(q0)
        qs = []
        for vel in q_targets:
            qs.append(dynamics.set_vel(vel).q)
        q_for_grad = th.stack(qs)
    else:
        rollout = dynamics.rollout(q_targets, q0=q0)
        q_for_grad = rollout.q
    ee = kin.end_effector_transform(q_for_grad[-1])
    loss = ee[:3, 3].square().sum()
    loss.backward()
    _assert(f"{mode}: differentiable target gradient", q_targets.grad is not None)
    _assert(f"{mode}: finite target gradient", th.isfinite(q_targets.grad).all())

    small_targets = th.stack(
        [th.tensor([0.10, -0.10, -0.08, 0.04, -0.04, 0.05], dtype=th.float64) for _ in range(20)]
    ).requires_grad_(True)
    if mode == "vel":
        dynamics.reset(q0)
        small_qs = []
        for vel in small_targets:
            small_qs.append(dynamics.set_vel(vel).q)
        small_q_for_grad = th.stack(small_qs)
    else:
        small_rollout = dynamics.rollout(small_targets, q0=q0)
        small_q_for_grad = small_rollout.q
    small_loss = kin.end_effector_transform(small_q_for_grad[-1])[:3, 3].square().sum()
    small_loss.backward()
    _assert(f"{mode}: finite unsaturated target gradient", th.isfinite(small_targets.grad).all())
    _assert(f"{mode}: nonzero unsaturated target gradient", small_targets.grad.abs().max() > 0)

    print(f"{mode}: ok")
    print(f"  final q: {[round(v, 4) for v in q.tolist()]}")
    print(f"  final dq: {[round(v, 4) for v in dq.tolist()]}")
    print(f"  vlim: {[round(v, 4) for v in vlim.tolist()]}")
    print(f"  grad max abs: {q_targets.grad.abs().max().item():.3e}")
    print(f"  unsaturated grad max abs: {small_targets.grad.abs().max().item():.3e}")


def main():
    for mode in ("pos_vel", "mit", "vel"):
        validate_mode(mode)


if __name__ == "__main__":
    main()
