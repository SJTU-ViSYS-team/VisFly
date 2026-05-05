import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch as th

sys.path.append(os.path.dirname(os.getcwd()))

from VisFly.utils.rebot import ReBotKinematics, ReBotRigidBodyDynamics  # noqa: E402


DEFAULT_URDF = "datasets/visfly-beta/urdf/rebot_devarm/reBot-DevArm_fixend.urdf"


def _load_pinocchio():
    try:
        import pinocchio as pin
    except ImportError as exc:
        raise SystemExit("Pinocchio is not installed. Install `pin` to run this comparison.") from exc
    return pin


def _sample(kin: ReBotKinematics, samples: int, seed: int):
    generator = th.Generator(device="cpu")
    generator.manual_seed(seed)
    lower, upper = kin.joint_limits
    q = lower + (upper - lower) * th.rand(samples, 6, dtype=th.float64, generator=generator)
    dq = 0.8 * (2.0 * th.rand(samples, 6, dtype=th.float64, generator=generator) - 1.0)
    ddq = 1.2 * (2.0 * th.rand(samples, 6, dtype=th.float64, generator=generator) - 1.0)
    return q, dq, ddq


def compare(args):
    pin = _load_pinocchio()
    kin = ReBotKinematics(args.urdf)
    dyn = ReBotRigidBodyDynamics(kinematics=kin)
    model = pin.buildModelFromUrdf(str(Path(args.urdf).resolve()))
    data = model.createData()

    qs, dqs, ddqs = _sample(kin, args.samples, args.seed)
    max_M = 0.0
    max_g = 0.0
    max_nle = 0.0
    max_tau = 0.0
    for q, dq, ddq in zip(qs, dqs, ddqs):
        q_np = q.numpy()
        dq_np = dq.numpy()
        ddq_np = ddq.numpy()

        model_M = dyn.mass_matrix(q).detach().numpy()
        model_g = dyn.gravity_vector(q).detach().numpy()
        model_nle = dyn.nonlinear_effects(q, dq).detach().numpy()
        model_tau = dyn.inverse_dynamics(q, dq, ddq).detach().numpy()

        pin.crba(model, data, q_np)
        pin_M = data.M.copy()
        pin.computeGeneralizedGravity(model, data, q_np)
        pin_g = data.g.copy()
        pin.nonLinearEffects(model, data, q_np, dq_np)
        pin_nle = data.nle.copy()
        pin_tau = pin.rnea(model, data, q_np, dq_np, ddq_np).copy()

        max_M = max(max_M, float(np.linalg.norm(model_M - pin_M, ord=np.inf)))
        max_g = max(max_g, float(np.linalg.norm(model_g - pin_g, ord=np.inf)))
        max_nle = max(max_nle, float(np.linalg.norm(model_nle - pin_nle, ord=np.inf)))
        max_tau = max(max_tau, float(np.linalg.norm(model_tau - pin_tau, ord=np.inf)))

    print(f"samples: {args.samples}")
    print(f"max_mass_matrix_error_inf: {max_M:.3e}")
    print(f"max_gravity_error_inf: {max_g:.3e}")
    print(f"max_nle_error_inf: {max_nle:.3e}")
    print(f"max_inverse_dynamics_error_inf: {max_tau:.3e}")
    if max_M > args.mass_tol or max_g > args.gravity_tol or max_nle > args.nle_tol or max_tau > args.tau_tol:
        raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare reBot rigid-body dynamics against Pinocchio."
    )
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--mass-tol", type=float, default=1e-8)
    parser.add_argument("--gravity-tol", type=float, default=1e-8)
    parser.add_argument("--nle-tol", type=float, default=1e-8)
    parser.add_argument("--tau-tol", type=float, default=1e-8)
    return parser.parse_args()


if __name__ == "__main__":
    compare(parse_args())
