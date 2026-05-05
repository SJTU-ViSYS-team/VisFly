from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import math
import xml.etree.ElementTree as ET

import torch as th


DEFAULT_REBOT_URDF = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "visfly-beta"
    / "urdf"
    / "rebot_devarm"
    / "reBot-DevArm_fixend.urdf"
)


@dataclass(frozen=True)
class ReBotModel:
    joint_names: List[str]
    link_names: List[str]
    lower_limits: th.Tensor
    upper_limits: th.Tensor
    joint_origins_xyz: th.Tensor
    joint_origins_rpy: th.Tensor
    joint_axes: th.Tensor
    end_origin_xyz: th.Tensor
    end_origin_rpy: th.Tensor
    link_masses: th.Tensor
    link_coms: th.Tensor
    link_inertias: th.Tensor


def _parse_vector(value: str, dtype: th.dtype = th.float64) -> th.Tensor:
    return th.tensor([float(v) for v in value.split()], dtype=dtype)


def _skew(v: th.Tensor) -> th.Tensor:
    zeros = th.zeros_like(v[..., 0])
    return th.stack(
        [
            th.stack([zeros, -v[..., 2], v[..., 1]], dim=-1),
            th.stack([v[..., 2], zeros, -v[..., 0]], dim=-1),
            th.stack([-v[..., 1], v[..., 0], zeros], dim=-1),
        ],
        dim=-2,
    )


def _rpy_to_matrix(rpy: th.Tensor) -> th.Tensor:
    roll, pitch, yaw = rpy.unbind(dim=-1)
    cr, sr = th.cos(roll), th.sin(roll)
    cp, sp = th.cos(pitch), th.sin(pitch)
    cy, sy = th.cos(yaw), th.sin(yaw)

    one = th.ones_like(roll)
    zero = th.zeros_like(roll)
    rx = th.stack(
        [
            th.stack([one, zero, zero], dim=-1),
            th.stack([zero, cr, -sr], dim=-1),
            th.stack([zero, sr, cr], dim=-1),
        ],
        dim=-2,
    )
    ry = th.stack(
        [
            th.stack([cp, zero, sp], dim=-1),
            th.stack([zero, one, zero], dim=-1),
            th.stack([-sp, zero, cp], dim=-1),
        ],
        dim=-2,
    )
    rz = th.stack(
        [
            th.stack([cy, -sy, zero], dim=-1),
            th.stack([sy, cy, zero], dim=-1),
            th.stack([zero, zero, one], dim=-1),
        ],
        dim=-2,
    )
    return rz @ ry @ rx


def _axis_angle_to_matrix(axis: th.Tensor, angle: th.Tensor) -> th.Tensor:
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    eye = th.eye(3, dtype=angle.dtype, device=angle.device)
    while eye.ndim < axis.ndim + 1:
        eye = eye.unsqueeze(0)
    eye = eye.expand(*axis.shape[:-1], 3, 3)

    k = _skew(axis)
    sin = th.sin(angle)[..., None, None]
    cos = th.cos(angle)[..., None, None]
    return eye + sin * k + (1.0 - cos) * (k @ k)


def _transform(rotation: th.Tensor, translation: th.Tensor) -> th.Tensor:
    T = th.zeros(*rotation.shape[:-2], 4, 4, dtype=rotation.dtype, device=rotation.device)
    T[..., :3, :3] = rotation
    T[..., :3, 3] = translation
    T[..., 3, 3] = 1.0
    return T


def _origin_to_transform(xyz: th.Tensor, rpy: th.Tensor) -> th.Tensor:
    return _transform(_rpy_to_matrix(rpy), xyz)


class ReBotKinematics:
    """Differentiable PyTorch kinematics for the reBot B601-DM fixed-end URDF.

    This class intentionally implements the fixed 6-revolute-joint serial chain
    needed by VisFly. Habitat is not involved, so FK/Jacobian outputs can be used
    in differentiable rewards and losses.
    """

    def __init__(
        self,
        urdf_path: str | Path = DEFAULT_REBOT_URDF,
        dtype: th.dtype = th.float64,
        device: th.device | str = "cpu",
    ):
        self.urdf_path = Path(urdf_path)
        self.dtype = dtype
        self.device = th.device(device)
        self.model = self._load_model(self.urdf_path)
        self.to(device=self.device, dtype=self.dtype)

    def to(
        self,
        device: Optional[th.device | str] = None,
        dtype: Optional[th.dtype] = None,
    ) -> "ReBotKinematics":
        device = self.device if device is None else th.device(device)
        dtype = self.dtype if dtype is None else dtype
        self.device = device
        self.dtype = dtype
        for name, value in self.model.__dict__.items():
            if isinstance(value, th.Tensor):
                object.__setattr__(self.model, name, value.to(device=device, dtype=dtype))
        return self

    @property
    def joint_names(self) -> List[str]:
        return self.model.joint_names

    @property
    def link_names(self) -> List[str]:
        return self.model.link_names

    @property
    def joint_limits(self) -> tuple[th.Tensor, th.Tensor]:
        return self.model.lower_limits, self.model.upper_limits

    def clamp(self, q: th.Tensor) -> th.Tensor:
        lower, upper = self.joint_limits
        return q.clamp(lower.to(q.device, q.dtype), upper.to(q.device, q.dtype))

    def forward_kinematics(
        self,
        q: th.Tensor,
        *,
        include_end_link: bool = True,
    ) -> Dict[str, th.Tensor]:
        q = self._normalize_q(q)
        batch_shape = q.shape[:-1]
        T = th.eye(4, dtype=q.dtype, device=q.device).expand(*batch_shape, 4, 4).clone()
        transforms: Dict[str, th.Tensor] = {}

        for i, link_name in enumerate(self.link_names[:-1]):
            origin = _origin_to_transform(
                self.model.joint_origins_xyz[i].to(q.device, q.dtype),
                self.model.joint_origins_rpy[i].to(q.device, q.dtype),
            ).expand(*batch_shape, 4, 4)
            axis = self.model.joint_axes[i].to(q.device, q.dtype).expand(*batch_shape, 3)
            rotation = _transform(
                _axis_angle_to_matrix(axis, q[..., i]),
                th.zeros(*batch_shape, 3, dtype=q.dtype, device=q.device),
            )
            T = T @ origin @ rotation
            transforms[link_name] = T

        if include_end_link:
            end_origin = _origin_to_transform(
                self.model.end_origin_xyz.to(q.device, q.dtype),
                self.model.end_origin_rpy.to(q.device, q.dtype),
            ).expand(*batch_shape, 4, 4)
            transforms[self.link_names[-1]] = T @ end_origin

        return transforms

    def end_effector_transform(self, q: th.Tensor) -> th.Tensor:
        return self.forward_kinematics(q, include_end_link=True)["end_link"]

    def geometric_jacobian(
        self,
        q: th.Tensor,
        link_name: str = "end_link",
    ) -> th.Tensor:
        return self.point_jacobian(q, link_name=link_name)

    def point_jacobian(
        self,
        q: th.Tensor,
        link_name: str = "end_link",
        point_local: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        q = self._normalize_q(q)
        batch_shape = q.shape[:-1]
        T = th.eye(4, dtype=q.dtype, device=q.device).expand(*batch_shape, 4, 4).clone()
        joint_positions = []
        joint_axes_world = []
        target_T = None

        active_joints = self._active_joint_count(link_name)
        for i, child_link in enumerate(self.link_names[:-1]):
            origin = _origin_to_transform(
                self.model.joint_origins_xyz[i].to(q.device, q.dtype),
                self.model.joint_origins_rpy[i].to(q.device, q.dtype),
            ).expand(*batch_shape, 4, 4)
            T_joint = T @ origin
            axis_local = self.model.joint_axes[i].to(q.device, q.dtype).expand(*batch_shape, 3)
            axis_world = T_joint[..., :3, :3] @ axis_local[..., :, None]
            joint_positions.append(T_joint[..., :3, 3])
            joint_axes_world.append(axis_world.squeeze(-1))

            rotation = _transform(
                _axis_angle_to_matrix(axis_local, q[..., i]),
                th.zeros(*batch_shape, 3, dtype=q.dtype, device=q.device),
            )
            T = T_joint @ rotation
            if child_link == link_name:
                target_T = T

        if link_name == "end_link":
            end_origin = _origin_to_transform(
                self.model.end_origin_xyz.to(q.device, q.dtype),
                self.model.end_origin_rpy.to(q.device, q.dtype),
            ).expand(*batch_shape, 4, 4)
            target_T = T @ end_origin

        if target_T is None:
            raise KeyError(f"Unknown reBot link: {link_name}")

        if point_local is None:
            p_target = target_T[..., :3, 3]
        else:
            point_local = th.as_tensor(point_local, dtype=q.dtype, device=q.device)
            p_target = target_T[..., :3, :3] @ point_local.expand(*batch_shape, 3)[..., :, None]
            p_target = p_target.squeeze(-1) + target_T[..., :3, 3]

        cols = []
        for i, (p_joint, axis) in enumerate(zip(joint_positions, joint_axes_world)):
            if i < active_joints:
                linear = th.linalg.cross(axis, p_target - p_joint, dim=-1)
                cols.append(th.cat([linear, axis], dim=-1))
            else:
                cols.append(th.zeros(*batch_shape, 6, dtype=q.dtype, device=q.device))
        return th.stack(cols, dim=-1)

    def _active_joint_count(self, link_name: str) -> int:
        if link_name == "end_link":
            return 6
        if link_name not in self.link_names:
            raise KeyError(f"Unknown reBot link: {link_name}")
        link_index = self.link_names.index(link_name)
        return min(link_index + 1, 6)

    def _normalize_q(self, q: th.Tensor) -> th.Tensor:
        q = th.as_tensor(q, dtype=self.dtype, device=self.device)
        if q.shape[-1] != 6:
            raise ValueError(f"Expected q.shape[-1] == 6, got {tuple(q.shape)}")
        return q

    def _load_model(self, urdf_path: Path) -> ReBotModel:
        if not urdf_path.exists():
            raise FileNotFoundError(f"reBot URDF not found: {urdf_path}")

        root = ET.parse(urdf_path).getroot()
        child_to_joint = {}
        for joint in root.findall("joint"):
            joint_type = joint.attrib["type"]
            child = joint.find("child").attrib["link"]
            parent = joint.find("parent").attrib["link"]
            origin = joint.find("origin")
            axis = joint.find("axis")
            limit = joint.find("limit")
            child_to_joint[child] = {
                "name": joint.attrib["name"],
                "type": joint_type,
                "parent": parent,
                "child": child,
                "xyz": _parse_vector(origin.attrib.get("xyz", "0 0 0")),
                "rpy": _parse_vector(origin.attrib.get("rpy", "0 0 0")),
                "axis": _parse_vector(axis.attrib.get("xyz", "0 0 1"))
                if axis is not None
                else th.tensor([0.0, 0.0, 1.0], dtype=th.float64),
                "lower": float(limit.attrib["lower"]) if limit is not None else -math.inf,
                "upper": float(limit.attrib["upper"]) if limit is not None else math.inf,
            }

        link = "base_link"
        moving_joints = []
        link_names = []
        while True:
            next_joint = None
            for joint in child_to_joint.values():
                if joint["parent"] == link:
                    next_joint = joint
                    break
            if next_joint is None:
                break
            if next_joint["type"] == "revolute":
                moving_joints.append(next_joint)
                link_names.append(next_joint["child"])
            elif next_joint["type"] == "fixed" and next_joint["child"] == "end_link":
                end_joint = next_joint
                link_names.append(next_joint["child"])
            else:
                raise ValueError(f"Unsupported reBot joint in chain: {next_joint}")
            link = next_joint["child"]

        if len(moving_joints) != 6:
            raise ValueError(f"Expected 6 revolute reBot joints, found {len(moving_joints)}")
        if "end_joint" not in locals():
            raise ValueError("Expected fixed end_joint ending at end_link")

        return ReBotModel(
            joint_names=[joint["name"] for joint in moving_joints],
            link_names=link_names,
            lower_limits=th.tensor([joint["lower"] for joint in moving_joints], dtype=th.float64),
            upper_limits=th.tensor([joint["upper"] for joint in moving_joints], dtype=th.float64),
            joint_origins_xyz=th.stack([joint["xyz"] for joint in moving_joints]),
            joint_origins_rpy=th.stack([joint["rpy"] for joint in moving_joints]),
            joint_axes=th.stack([joint["axis"] for joint in moving_joints]),
            end_origin_xyz=end_joint["xyz"],
            end_origin_rpy=end_joint["rpy"],
            link_masses=self._parse_link_masses(root, link_names),
            link_coms=self._parse_link_coms(root, link_names),
            link_inertias=self._parse_link_inertias(root, link_names),
        )

    @staticmethod
    def _parse_link_masses(root: ET.Element, link_names: Iterable[str]) -> th.Tensor:
        masses = []
        for name in link_names:
            link = root.find(f"link[@name='{name}']")
            inertial = link.find("inertial") if link is not None else None
            mass = inertial.find("mass") if inertial is not None else None
            masses.append(float(mass.attrib["value"]) if mass is not None else 0.0)
        return th.tensor(masses, dtype=th.float64)

    @staticmethod
    def _parse_link_coms(root: ET.Element, link_names: Iterable[str]) -> th.Tensor:
        coms = []
        for name in link_names:
            link = root.find(f"link[@name='{name}']")
            inertial = link.find("inertial") if link is not None else None
            origin = inertial.find("origin") if inertial is not None else None
            coms.append(_parse_vector(origin.attrib.get("xyz", "0 0 0")) if origin is not None else th.zeros(3, dtype=th.float64))
        return th.stack(coms)

    @staticmethod
    def _parse_link_inertias(root: ET.Element, link_names: Iterable[str]) -> th.Tensor:
        inertias = []
        for name in link_names:
            link = root.find(f"link[@name='{name}']")
            inertial = link.find("inertial") if link is not None else None
            inertia = inertial.find("inertia") if inertial is not None else None
            if inertia is None:
                inertias.append(th.zeros(3, 3, dtype=th.float64))
                continue
            ixx = float(inertia.attrib["ixx"])
            ixy = float(inertia.attrib["ixy"])
            ixz = float(inertia.attrib["ixz"])
            iyy = float(inertia.attrib["iyy"])
            iyz = float(inertia.attrib["iyz"])
            izz = float(inertia.attrib["izz"])
            inertias.append(
                th.tensor(
                    [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]],
                    dtype=th.float64,
                )
            )
        return th.stack(inertias)
