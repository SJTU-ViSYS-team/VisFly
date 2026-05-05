from __future__ import annotations

from typing import Optional

import torch as th

from .kinematics import ReBotKinematics


class ReBotRigidBodyDynamics:
    """Differentiable rigid-body dynamics terms for the fixed reBot URDF.

    The implementation uses link COM Jacobians and PyTorch autograd rather than
    a hand-written RNEA/CRBA. This is slower than Pinocchio, but concise,
    differentiable, and straightforward to validate:

        tau = M(q) ddq + C(q, dq) dq + g(q)

    Contact forces and motor electrical dynamics are intentionally out of scope.
    """

    def __init__(
        self,
        kinematics: Optional[ReBotKinematics] = None,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        dtype: th.dtype = th.float64,
        device: th.device | str = "cpu",
    ):
        self.kinematics = kinematics if kinematics is not None else ReBotKinematics(dtype=dtype, device=device)
        self.dtype = dtype
        self.device = th.device(device)
        self.gravity = th.tensor(gravity, dtype=dtype, device=self.device)

    def to(self, device: th.device | str, dtype: Optional[th.dtype] = None) -> "ReBotRigidBodyDynamics":
        self.device = th.device(device)
        self.dtype = self.dtype if dtype is None else dtype
        self.kinematics.to(device=self.device, dtype=self.dtype)
        self.gravity = self.gravity.to(device=self.device, dtype=self.dtype)
        return self

    def mass_matrix(self, q: th.Tensor) -> th.Tensor:
        if q.ndim == 1:
            return self._mass_matrix_single(q)
        return th.stack([self._mass_matrix_single(q_i) for q_i in q], dim=0)

    def gravity_vector(self, q: th.Tensor) -> th.Tensor:
        if q.ndim == 1:
            return self._gravity_single(q)
        return th.stack([self._gravity_single(q_i) for q_i in q], dim=0)

    def nonlinear_effects(self, q: th.Tensor, dq: th.Tensor) -> th.Tensor:
        ddq = th.zeros_like(dq)
        return self.inverse_dynamics(q, dq, ddq)

    def inverse_dynamics(self, q: th.Tensor, dq: th.Tensor, ddq: th.Tensor) -> th.Tensor:
        if q.ndim == 1:
            return self._inverse_dynamics_single(q, dq, ddq)
        return th.stack(
            [self._inverse_dynamics_single(q_i, dq_i, ddq_i) for q_i, dq_i, ddq_i in zip(q, dq, ddq)],
            dim=0,
        )

    def forward_dynamics(self, q: th.Tensor, dq: th.Tensor, tau: th.Tensor) -> th.Tensor:
        M = self.mass_matrix(q)
        nle = self.nonlinear_effects(q, dq)
        return th.linalg.solve(M, (tau - nle).unsqueeze(-1)).squeeze(-1)

    def kinetic_energy(self, q: th.Tensor, dq: th.Tensor) -> th.Tensor:
        M = self.mass_matrix(q)
        return 0.5 * (dq.unsqueeze(-2) @ M @ dq.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    def potential_energy(self, q: th.Tensor) -> th.Tensor:
        if q.ndim == 1:
            return self._potential_energy_single(q)
        return th.stack([self._potential_energy_single(q_i) for q_i in q], dim=0)

    def _mass_matrix_single(self, q: th.Tensor) -> th.Tensor:
        q = th.as_tensor(q, dtype=self.dtype, device=self.device)
        transforms = self.kinematics.forward_kinematics(q, include_end_link=True)
        M = th.zeros(6, 6, dtype=q.dtype, device=q.device)
        for idx, link_name in enumerate(self.kinematics.link_names):
            mass = self.kinematics.model.link_masses[idx].to(q.device, q.dtype)
            com = self.kinematics.model.link_coms[idx].to(q.device, q.dtype)
            inertia_local = self.kinematics.model.link_inertias[idx].to(q.device, q.dtype)
            J = self.kinematics.point_jacobian(q, link_name=link_name, point_local=com)
            Jv, Jw = J[:3], J[3:]
            R = transforms[link_name][:3, :3]
            inertia_world = R @ inertia_local @ R.T
            M = M + mass * (Jv.T @ Jv) + Jw.T @ inertia_world @ Jw
        return 0.5 * (M + M.T)

    def _potential_energy_single(self, q: th.Tensor) -> th.Tensor:
        q = th.as_tensor(q, dtype=self.dtype, device=self.device)
        transforms = self.kinematics.forward_kinematics(q, include_end_link=True)
        potential = th.zeros((), dtype=q.dtype, device=q.device)
        gravity = self.gravity.to(q.device, q.dtype)
        for idx, link_name in enumerate(self.kinematics.link_names):
            mass = self.kinematics.model.link_masses[idx].to(q.device, q.dtype)
            com = self.kinematics.model.link_coms[idx].to(q.device, q.dtype)
            T = transforms[link_name]
            p_com = T[:3, :3] @ com + T[:3, 3]
            potential = potential - mass * gravity.dot(p_com)
        return potential

    def _gravity_single(self, q: th.Tensor) -> th.Tensor:
        q_req = th.as_tensor(q, dtype=self.dtype, device=self.device)
        if not q_req.requires_grad:
            q_req = q_req.clone().requires_grad_(True)
        U = self._potential_energy_single(q_req)
        return th.autograd.grad(U, q_req, create_graph=True)[0]

    def _inverse_dynamics_single(self, q: th.Tensor, dq: th.Tensor, ddq: th.Tensor) -> th.Tensor:
        q_req = th.as_tensor(q, dtype=self.dtype, device=self.device)
        dq_req = th.as_tensor(dq, dtype=self.dtype, device=self.device)
        ddq = th.as_tensor(ddq, dtype=self.dtype, device=self.device)
        if not q_req.requires_grad:
            q_req = q_req.clone().requires_grad_(True)
        if not dq_req.requires_grad:
            dq_req = dq_req.clone().requires_grad_(True)

        K = self.kinetic_energy(q_req, dq_req)
        dK_ddq = th.autograd.grad(K, dq_req, create_graph=True, retain_graph=True)[0]
        dK_dq = th.autograd.grad(K, q_req, create_graph=True, retain_graph=True)[0]

        dp_dq = th.stack(
            [th.autograd.grad(dK_ddq[i], q_req, create_graph=True, retain_graph=True)[0] for i in range(6)],
            dim=0,
        )
        dp_ddq = th.stack(
            [th.autograd.grad(dK_ddq[i], dq_req, create_graph=True, retain_graph=True)[0] for i in range(6)],
            dim=0,
        )
        gravity = self._gravity_single(q_req)
        return dp_dq @ dq_req + dp_ddq @ ddq - dK_dq + gravity
