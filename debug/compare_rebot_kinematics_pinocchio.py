import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch as th

sys.path.append(os.path.dirname(os.getcwd()))

from VisFly.utils.rebot import ReBotKinematics  # noqa: E402


DEFAULT_URDF = "datasets/visfly-beta/urdf/rebot_devarm/reBot-DevArm_fixend.urdf"


def _load_pinocchio():
    try:
        import pinocchio as pin
    except ImportError as exc:
        raise SystemExit(
            "Pinocchio is not installed in this environment. Install the upstream "
            "dependency package, for example `python -m pip install pin`, then rerun."
        ) from exc
    return pin


def _sample_q(kin: ReBotKinematics, num_samples: int, seed: int) -> th.Tensor:
    generator = th.Generator(device="cpu")
    generator.manual_seed(seed)
    lower, upper = kin.joint_limits
    lower = lower.cpu()
    upper = upper.cpu()
    return lower + (upper - lower) * th.rand(
        num_samples, 6, dtype=th.float64, generator=generator
    )


def _pin_fk_and_jacobian(pin, model, data, q_np, frame_id):
    pin.forwardKinematics(model, data, q_np)
    pin.updateFramePlacements(model, data)
    T = data.oMf[frame_id]
    pin.computeJointJacobians(model, data, q_np)
    # ReBotKinematics.geometric_jacobian returns [linear; angular] with both
    # components expressed in the world axes and linear velocity measured at the
    # target frame origin. This matches Pinocchio LOCAL_WORLD_ALIGNED.
    jac = pin.getFrameJacobian(model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    matrix = np.eye(4)
    matrix[:3, :3] = T.rotation
    matrix[:3, 3] = T.translation
    return matrix, jac


def compare(args):
    pin = _load_pinocchio()
    kin = ReBotKinematics(args.urdf)
    model = pin.buildModelFromUrdf(str(Path(args.urdf).resolve()))
    data = model.createData()
    frame_id = model.getFrameId(args.frame)
    if frame_id == len(model.frames):
        raise KeyError(f"Pinocchio frame not found: {args.frame}")

    qs = _sample_q(kin, args.samples, args.seed)
    max_translation_error = 0.0
    max_rotation_error = 0.0
    max_jacobian_error = 0.0

    for q in qs:
        model_T = kin.end_effector_transform(q).detach().cpu().numpy()
        model_J = kin.geometric_jacobian(q, args.frame).detach().cpu().numpy()
        pin_T, pin_J = _pin_fk_and_jacobian(pin, model, data, q.detach().cpu().numpy(), frame_id)

        translation_error = np.linalg.norm(model_T[:3, 3] - pin_T[:3, 3], ord=np.inf)
        rotation_error = np.linalg.norm(model_T[:3, :3] - pin_T[:3, :3], ord=np.inf)
        jacobian_error = np.linalg.norm(model_J - pin_J, ord=np.inf)

        max_translation_error = max(max_translation_error, float(translation_error))
        max_rotation_error = max(max_rotation_error, float(rotation_error))
        max_jacobian_error = max(max_jacobian_error, float(jacobian_error))

    print(f"samples: {args.samples}")
    print(f"joint_names: {kin.joint_names}")
    print(f"max_translation_error_inf: {max_translation_error:.3e}")
    print(f"max_rotation_error_inf: {max_rotation_error:.3e}")
    print(f"max_jacobian_error_inf: {max_jacobian_error:.3e}")

    if (
        max_translation_error > args.translation_tol
        or max_rotation_error > args.rotation_tol
        or max_jacobian_error > args.jacobian_tol
    ):
        raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare reBot FK/Jacobian against Pinocchio."
    )
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--frame", default="end_link")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--translation-tol", type=float, default=1e-9)
    parser.add_argument("--rotation-tol", type=float, default=1e-9)
    parser.add_argument("--jacobian-tol", type=float, default=1e-8)
    return parser.parse_args()


if __name__ == "__main__":
    compare(parse_args())
