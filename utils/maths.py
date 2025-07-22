import torch as th


class Quaternion:
    def __init__(self, w=None, x=None, y=None, z=None, num=1, device=th.device("cpu")):
        assert type(w) == type(x) == type(y) == type(z)  # "w, x, y, z should have the same type"
        if w is None:
            self.w = th.ones(num, device=device)
            self.x = th.zeros(num, device=device)
            self.y = th.zeros(num, device=device)
            self.z = th.zeros(num, device=device)
        elif isinstance(w, (int, float)):
            self.w = th.ones(num, device=device) * w
            self.x = th.ones(num, device=device) * x
            self.y = th.ones(num, device=device) * y
            self.z = th.ones(num, device=device) * z
        elif isinstance(w, th.Tensor):
            self.w = w
            self.x = x
            self.y = y
            self.z = z
        else:
            raise ValueError("unsupported type")

    def to(self, device):
        self.w = self.w.to(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.z = self.z.to(device)
        return self

    def rotate(self, other):
        if isinstance(other, Quaternion):
            # quaternion multiplication
            return self * other
        elif other.shape[0] == 3:
            # vector rotation
            return (self * Quaternion(th.tensor(0), *other) * self.conjugate()).imag

    def inv_rotate(self, other):
        if isinstance(other, Quaternion):
            # quaternion multiplication
            return self.conjugate() * other
        elif other.shape[0] == 3:
            # vector rotation
            return (self.conjugate() * Quaternion(th.tensor(0), *other) * self).imag

    def transform(self, other):
        return self.inv_rotate(other)

    def inv_transform(self, other):
        return self.rotate(other)

    @property
    def R(self):
        # return th.permute(th.stack([
        #     th.stack([1 - 2 * (self.y.pow(2) + self.z.pow(2)), 2 * (self.x * self.y - self.z * self.w), 2 * (self.x * self.z + self.y * self.w)]),
        #     th.stack([2 * (self.x * self.y + self.z * self.w), 1 - 2 * (self.x.pow(2) + self.z.pow(2)), 2 * (self.y * self.z - self.x * self.w)]),
        #     th.stack([2 * (self.x * self.z - self.y * self.w), 2 * (self.y * self.z + self.x * self.w), 1 - 2 * (self.x.pow(2) + self.y.pow(2))])
        # ]), (2,0,1))
        return th.stack([
            th.stack([1 - 2 * (self.y.pow(2) + self.z.pow(2)), 2 * (self.x * self.y - self.z * self.w), 2 * (self.x * self.z + self.y * self.w)]),
            th.stack([2 * (self.x * self.y + self.z * self.w), 1 - 2 * (self.x.pow(2) + self.z.pow(2)), 2 * (self.y * self.z - self.x * self.w)]),
            th.stack([2 * (self.x * self.z - self.y * self.w), 2 * (self.y * self.z + self.x * self.w), 1 - 2 * (self.x.pow(2) + self.y.pow(2))])
        ])

    @property
    def x_axis(self):
        # return th.stack([1 - 2 * (self.y.pow(2) + self.z.pow(2)), 2 * (self.x * self.y + self.z * self.w), 2 * (self.x * self.z - self.y * self.w)])
        x_axis = th.stack([
            # 1 - 2 * (self.y*self.y + self.z*self.z),
            #  2 * (self.x * self.y + self.z * self.w),
            # 2 * (self.x * self.z - self.y * self.w)
            1 - 2 * (self.y.clone() * self.y.clone() + self.z.clone() * self.z.clone()),
            2 * (self.x.clone() * self.y.clone() + self.z.clone() * self.w.clone()),
            2 * (self.x.clone() * self.z.clone() - self.y.clone() * self.w.clone())
        ])
        return x_axis
    @property
    def xz_axis(self):
        # return th.stack([
        #     th.stack([1 - 2 * (self.y.pow(2) + self.z.pow(2)),
        #     2 * (self.x * self.y - self.z * self.w),
        #     2 * (self.x * self.z + self.y * self.w)]),
        #     th.stack([2 * (self.x * self.z + self.y * self.w),
        #     2 * (self.y * self.z - self.x * self.w),
        #     1 - 2 * (self.x.pow(2) + self.y.pow(2))])
        # ]) # debug xzaxis format using clone like x_axis
        return th.stack([
            th.stack([1 - 2 * (self.y.clone() * self.y.clone() + self.z.clone() * self.z.clone()),
                        2 * (self.x.clone() * self.y.clone() - self.z.clone() * self.w.clone()),
                        2 * (self.x.clone() * self.z.clone() + self.y.clone() * self.w.clone())]),
            th.stack([2 * (self.x.clone() * self.z.clone() + self.y.clone() * self.w.clone()),
                        2 * (self.y.clone() * self.z.clone() - self.x.clone() * self.w.clone()),
                        1 - 2 * (self.x.clone() * self.x.clone() + self.y.clone() * self.y.clone())])
        ])




    @property
    def shape(self):
        return 4, len(self)

    @property
    def real(self):
        return self.w

    @property
    def imag(self):
        return th.stack([self.x, self.y, self.z])

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)
        elif isinstance(other, (int, float, th.Tensor)):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        else:
            raise ValueError("unsupported type")

    def __truediv__(self, other):
        if isinstance(other, (int, float, th.Tensor)):
            return Quaternion(self.w / other, self.x / other, self.y / other, self.z / other)
        else:
            raise ValueError("unsupported type")

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, th.Tensor):
            return Quaternion(self.w + other[0], self.x + other[1], self.y + other[2], self.z + other[3])
        else:
            raise ValueError("unsupported type")

    def __sub__(self, other):
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self):
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __repr__(self):
        return f'({self.w}, {self.x}i, {self.y}j, {self.z}k)'

    def __getitem__(self, indices):
        return Quaternion(self.w[indices], self.x[indices], self.y[indices], self.z[indices])

    def __setitem__(self, indices, value):
        if isinstance(value, Quaternion):
            # Assign directly for a single index
            self.w[indices] = value.w
            self.x[indices] = value.x
            self.y[indices] = value.y
            self.z[indices] = value.z
        elif isinstance(value, th.Tensor):
            # Assign directly for a single index

            self.w[indices] = value[0]
            self.x[indices] = value[1]
            self.y[indices] = value[2]
            self.z[indices] = value[3]
        else:
            raise ValueError("Assigned value must be an instance of quaternion")

    def inverse(self):
        return self.conjugate() / self.norm()

    def norm(self):
        return th.sqrt(self.w.pow(2) + self.x.pow(2) + self.y.pow(2) + self.z.pow(2))

    def normalize(self):
        return self / self.norm()

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def toTensor(self):
        return th.stack([self.w, self.x, self.y, self.z])

    def append(self, other):
        self.w = th.cat([self.w, other.w])
        self.x = th.cat([self.x, other.x])
        self.y = th.cat([self.y, other.y])
        self.z = th.cat([self.z, other.z])

    def toEuler(self, order="zyx"):
        if order == "zyx":
            roll = th.atan2(2 * (self.w * self.x + self.y * self.z), 1 - 2 * (self.x.pow(2) + self.y.pow(2)))
            pitch = th.asin(2 * (self.w * self.y - self.z * self.x))
            yaw = th.atan2(2 * (self.w * self.z + self.x * self.y), 1 - 2 * (self.y.pow(2) + self.z.pow(2)))
            return th.stack([roll, pitch, yaw])  # roll pitch yaw
        elif order == "xyz":
            roll = th.atleast_1d(th.atan2(2 * (self.w * self.y - self.x * self.z), 1 - 2 * (self.x.pow(2) + self.y.pow(2))))
            pitch = th.atleast_1d(th.asin(2 * (self.w * self.z - self.y * self.x)))
            yaw = th.atleast_1d(th.atan2(2 * (self.w * self.x + self.y * self.z), 1 - 2 * (self.x.pow(2) + self.z.pow(2))))
            return th.stack([roll, pitch, yaw])  # roll pitch yaw

    @staticmethod
    def from_euler(roll, pitch, yaw, order="zyx"):
        roll, pitch, yaw = th.as_tensor(roll), th.as_tensor(pitch), th.as_tensor(yaw)
        if order == "zyx":
            cy = th.cos(yaw * 0.5)
            sy = th.sin(yaw * 0.5)
            cp = th.cos(pitch * 0.5)
            sp = th.sin(pitch * 0.5)
            cr = th.cos(roll * 0.5)
            sr = th.sin(roll * 0.5)
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy
        elif order == "xyz":
            cy = th.cos(yaw * 0.5)
            sy = th.sin(yaw * 0.5)
            cp = th.cos(pitch * 0.5)
            sp = th.sin(pitch * 0.5)
            cr = th.cos(roll * 0.5)
            sr = th.sin(roll * 0.5)
            w = cr * cp * cy - sr * sp * sy
            x = sr * cp * cy + cr * sp * sy
            y = cr * sp * cy - sr * cp * sy
            z = cr * cp * sy + sr * sp * cy
        return Quaternion(w, x, y, z)

    def clone(self):
        return Quaternion(self.w.clone(), self.x.clone(), self.y.clone(), self.z.clone())

    def detach(self):
        return Quaternion(self.w.detach(), self.x.detach(), self.y.detach(), self.z.detach())

    def __len__(self):
        try:
            return len(self.w)
        except TypeError:
            return 1


class Integrator:
    def __init__(self):
        order = 1

    @staticmethod
    def _get_derivatives(vel: th.tensor,
                         ori: th.tensor,
                         acc: th.tensor,
                         ori_vel: th.tensor,
                         tau: th.tensor,
                         J: th.tensor,
                         J_inv: th.tensor):
        d_pos = vel
        d_q = (ori * Quaternion(th.tensor(0), *ori_vel) * 0.5).toTensor()
        d_vel = acc
        # d_ori_vel = J_inv @ (tau - ori_vel.cross(J @ ori_vel))
        d_ori_vel = J_inv @ (tau - th.linalg.cross(ori_vel.T, (J @ ori_vel).T).T)
        return d_pos, d_q, d_vel, d_ori_vel

    @staticmethod
    def integrate(
            pos: th.tensor,
            ori: th.tensor,
            vel: th.tensor,
            ori_vel: th.tensor,
            acc: th.tensor,
            tau: th.tensor,
            J: th.tensor,
            J_inv: th.tensor,
            dt: th.tensor,
            type="euler"
    ):
        if type == "euler":
            pos_cache, ori_cache, vel_cache, ori_vel_cache = pos.clone(), ori.clone(), vel.clone(), ori_vel.clone()

            d_pos, d_ori, d_vel, d_ori_vel = Integrator._get_derivatives(
                vel=vel_cache,
                ori=ori_cache,
                acc=acc,
                ori_vel=ori_vel_cache,
                tau=tau,
                J=J,
                J_inv=J_inv
            )
            pos += d_pos * dt
            ori += d_ori * dt
            vel += d_vel * dt
            ori_vel += d_ori_vel * dt

            # ori = ori / ori.norm()

            return pos, ori, vel, ori_vel, d_ori_vel

        elif type == "rk4":
            ks = th.tensor([1., 2., 2., 1.]) / 6
            slice_ts = th.tensor([0.5, 0.5, 1])
            pos_cache, ori_cache, vel_cache, ori_vel_cache = \
                pos.clone(), ori.clone(), vel.clone(), ori_vel.clone()
            d_pos = th.zeros((pos.shape[0], pos.shape[1], 4))
            d_ori = th.zeros((ori.shape[0], ori.shape[1], 4))
            d_vel = th.zeros((vel.shape[0], vel.shape[1], 4))
            d_ori_vel = th.zeros((ori_vel.shape[0], ori_vel.shape[1], 4))

            for index in range(4):
                # pos_cache = pos + d_pos * slice_ts[index] * dt
                if index != 0:
                    ori_cache = ori + d_ori[:, :, index - 1] * slice_ts[index - 1] * dt
                    vel_cache = vel + d_vel[:, :, index - 1] * slice_ts[index - 1] * dt
                    ori_vel_cache = ori_vel + d_ori_vel[:, :, index - 1] * slice_ts[index - 1] * dt

                d_pos[:, :, index], d_ori[:, :, index], d_vel[:, :, index], d_ori_vel[:, :, index] = \
                    Integrator._get_derivatives(
                        vel=vel_cache,
                        ori=ori_cache,
                        acc=acc,
                        ori_vel=ori_vel_cache,
                        tau=tau,
                        J=J,
                        J_inv=J_inv
                    )
            # f"w_cache: {ori_vel_cache} quat:{ori_cache} d_ori:{d_ori[:,:,index]}"
            pos += d_pos @ ks * dt
            ori += d_ori @ ks * dt
            vel += d_vel @ ks * dt
            ori_vel += d_ori_vel @ ks * dt

            return pos, ori, vel, ori_vel, d_ori_vel

        else:
            raise ValueError("type should be one of ['euler', 'rk4']")


def cross(a: th.Tensor, b: th.Tensor):
    res = th.stack([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]) + 0
    return res


def debug():
    test = 1


if __name__ == "__main__":
    debug()
