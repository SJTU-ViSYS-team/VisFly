import os
import sys

import torch as th

sys.path.append(os.path.dirname(os.getcwd()))

from VisFly.utils.rebot import ReBotGripperDynamics  # noqa: E402


def _assert(name, condition):
    if not bool(condition):
        raise AssertionError(name)


def validate_mode(mode: str):
    gripper = ReBotGripperDynamics(mode=mode, dt=0.01, position_limit=(-0.8, 0.8))
    gripper.reset(q=0.0)
    limit = gripper.config.velocity_limit

    for _ in range(100):
        if mode == "pos_vel":
            state = gripper.pos_vel(2.0)
        elif mode == "mit":
            state = gripper.mit(2.0)
        elif mode == "vel":
            state = gripper.set_vel(2.0)
        else:
            raise RuntimeError(mode)

    _assert(f"{mode}: q lower limit", state.q >= gripper.position_limit[0] - 1e-12)
    _assert(f"{mode}: q upper limit", state.q <= gripper.position_limit[1] + 1e-12)
    _assert(f"{mode}: dq velocity limit", state.dq.abs() <= limit + 1e-12)

    targets = th.linspace(0.0, 0.2, 20, dtype=th.float64, requires_grad=True)
    gripper.reset(q=0.0)
    states = []
    for target in targets:
        if mode == "pos_vel":
            states.append(gripper.pos_vel(target).q)
        elif mode == "mit":
            states.append(gripper.mit(target).q)
        elif mode == "vel":
            states.append(gripper.set_vel(target).q)
    loss = th.stack(states).square().sum()
    loss.backward()
    _assert(f"{mode}: differentiable target gradient", targets.grad is not None)
    _assert(f"{mode}: finite target gradient", th.isfinite(targets.grad).all())

    print(f"{mode}: ok")
    print(f"  final q: {state.q.item():+.4f}")
    print(f"  final dq: {state.dq.item():+.4f}")
    print(f"  vlim: {limit.item():.4f}")
    print(f"  grad max abs: {targets.grad.abs().max().item():.3e}")


def main():
    for mode in ("pos_vel", "mit", "vel"):
        validate_mode(mode)


if __name__ == "__main__":
    main()
