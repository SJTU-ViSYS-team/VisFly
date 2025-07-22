import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, splprep, splev
import torch as th


class SmoothPathGenerator:
    def __init__(self, control_points, method='cubic_spline', num_points=1000):
        """
        光滑路径生成器

        Args:
            control_points: 控制点列表 [[x1,y1,z1], [x2,y2,z2], ...]
            method: 插值方法 ('cubic_spline', 'b_spline', 'bezier')
            num_points: 生成的路径点数量
        """
        self.control_points = np.array(control_points)
        self.method = method
        self.num_points = num_points

    def generate_path(self):
        """生成光滑路径"""
        if self.method == 'cubic_spline':
            return self._cubic_spline_interpolation()
        elif self.method == 'b_spline':
            return self._b_spline_interpolation()
        elif self.method == 'bezier':
            return self._bezier_interpolation()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _cubic_spline_interpolation(self):
        """三次样条插值"""
        # 计算累积弧长作为参数
        distances = np.sqrt(np.sum(np.diff(self.control_points, axis=0) ** 2, axis=1))
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])

        # 为每个维度创建三次样条
        cs_x = CubicSpline(cumulative_distances, self.control_points[:, 0], bc_type="clamped")
        cs_y = CubicSpline(cumulative_distances, self.control_points[:, 1], bc_type="clamped")
        cs_z = CubicSpline(cumulative_distances, self.control_points[:, 2], bc_type="clamped")
        cs_x = CubicSpline(cumulative_distances, self.control_points[:, 0], bc_type="not-a-knot")
        cs_y = CubicSpline(cumulative_distances, self.control_points[:, 1], bc_type="not-a-knot")
        cs_z = CubicSpline(cumulative_distances, self.control_points[:, 2], bc_type="not-a-knot")

        # 生成插值点
        t = np.linspace(0, cumulative_distances[-1], self.num_points)
        path = np.column_stack([cs_x(t), cs_y(t), cs_z(t)])

        return path, t

    def _b_spline_interpolation(self):
        """B样条插值"""
        # 使用scipy的参数化B样条
        tck, u = splprep([self.control_points[:, 0],
                          self.control_points[:, 1],
                          self.control_points[:, 2]], s=0)

        # 生成插值点
        u_new = np.linspace(0, 1, self.num_points)
        path = np.column_stack(splev(u_new, tck))

        return path, u_new

    def _bezier_interpolation(self):
        """贝塞尔曲线插值"""

        def bezier_curve(control_points, t):
            """计算贝塞尔曲线上的点"""
            n = len(control_points) - 1
            result = np.zeros(3)

            for i, point in enumerate(control_points):
                # 二项式系数
                binomial_coeff = np.math.comb(n, i)
                # 贝塞尔基函数
                basis = binomial_coeff * (t ** i) * ((1 - t) ** (n - i))
                result += basis * point

            return result

        # 生成贝塞尔曲线
        t = np.linspace(0, 1, self.num_points)
        path = np.array([bezier_curve(self.control_points, ti) for ti in t])

        return path, t


class SmoothPath:
    """可以集成到你的Path类中的光滑路径"""

    def __init__(self, control_points, velocity, method='cubic_spline'):
        self.control_points = np.array(control_points)
        self.velocity = velocity
        self.method = method

        # 生成光滑路径
        generator = SmoothPathGenerator(control_points, method)
        self.path_points, self.path_params = generator.generate_path()

        # 计算路径总长度
        self.path_lengths = np.sqrt(np.sum(np.diff(self.path_points, axis=0) ** 2, axis=1))
        self.cumulative_lengths = np.concatenate([[0], np.cumsum(self.path_lengths)])
        self.total_length = self.cumulative_lengths[-1]

        # 当前状态
        self.current_distance = 0.0

    def get_target(self, t, pos, dt):
        """获取目标位置"""
        # 根据速度更新当前距离
        self.current_distance += self.velocity * dt

        # 如果超过路径长度，循环或停止
        if self.current_distance >= self.total_length:
            self.current_distance = self.total_length  # 停止在终点
            # 或者循环: self.current_distance = 0.0

        # 根据距离找到对应的路径点
        idx = np.searchsorted(self.cumulative_lengths, self.current_distance)
        if idx >= len(self.path_points):
            idx = len(self.path_points) - 1

        target_point = self.path_points[idx]
        return th.tensor(target_point, dtype=th.float32)

    def reset(self):
        """重置路径"""
        self.current_distance = 0.0


def demo_smooth_paths():
    """演示不同的光滑路径方法"""
    # 定义5个控制点
    control_points = [
        [0, 0, 0],  # 起点
        [3, 5, 2],  # 控制点1
        [8, 3, 5],  # 控制点2
        [12, 8, 3],  # 控制点3
        [15, 2, 1]  # 终点
    ]

    # 创建不同方法的路径生成器
    methods = ['cubic_spline', 'b_spline', 'bezier']

    fig = plt.figure(figsize=(15, 10))

    for i, method in enumerate(methods):
        # 生成路径
        generator = SmoothPathGenerator(control_points, method=method, num_points=200)
        path, params = generator.generate_path()

        # 3D图
        ax_3d = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax_3d.plot(path[:, 0], path[:, 1], path[:, 2], 'b-', linewidth=2, label=f'{method} path')
        ax_3d.scatter(*np.array(control_points).T, color='red', s=100, label='Control Points')

        # 标注控制点
        for j, point in enumerate(control_points):
            ax_3d.text(point[0], point[1], point[2], f'P{j}', fontsize=10)

        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title(f'3D {method.replace("_", " ").title()}')
        ax_3d.legend()

        # XY投影图
        ax_2d = fig.add_subplot(2, 3, i + 4)
        ax_2d.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label=f'{method} path')
        ax_2d.scatter(*np.array(control_points)[:, :2].T, color='red', s=100, label='Control Points')

        # 标注控制点
        for j, point in enumerate(control_points):
            ax_2d.text(point[0], point[1], f'P{j}', fontsize=10)

        ax_2d.set_xlabel('X')
        ax_2d.set_ylabel('Y')
        ax_2d.set_title(f'XY Projection - {method.replace("_", " ").title()}')
        ax_2d.grid(True, alpha=0.3)
        ax_2d.legend()
        ax_2d.axis('equal')

    plt.tight_layout()
    plt.show()


# 修改你的Path类以支持smooth_path
class Path:
    def __init__(self, cls, velocity, kwargs):
        self.cls = cls
        self.velocity = velocity
        if "comment" in kwargs:
            kwargs.pop("comment")

        if cls == "circle":
            self.radius = kwargs["radius"]
            self.center = kwargs["center"]
            self.angular_velocity = self.velocity / self.radius
        elif cls == "polygon":
            self.points = th.as_tensor(kwargs["points"])
            self.num_points = len(self.points)
            assert self.num_points >= 2, "Polygon path must have at least two points."
            self.current_index = 0
            self.position = self.points[0]
        elif cls == "smooth_path":
            # 新增的光滑路径类型
            control_points = kwargs["control_points"]
            method = kwargs.get("method", "cubic_spline")
            self.smooth_path = SmoothPath(control_points, velocity, method)

    def get_target(self, t, pos, dt):
        if self.cls == "circle":
            x = self.radius * np.cos(self.angular_velocity * t) + self.center[0]
            y = self.radius * np.sin(self.angular_velocity * t) + self.center[1]
            z = self.center[2]
            return th.tensor([x, y, z], dtype=th.float32).unsqueeze(0)
        elif self.cls == "polygon":
            next_index = (self.current_index + 1) % self.num_points
            dis_vec = self.points[next_index] - self.position
            dis_norm = dis_vec.norm()
            dis_vector = dis_vec / dis_norm
            velocity = self.velocity if dis_norm > self.velocity * dt else dis_norm / dt
            self.position = self.position + dis_vector * velocity * dt

            if dis_norm <= self.velocity * dt:
                self.current_index = next_index
            return self.position
        elif self.cls == "smooth_path":
            # 使用光滑路径
            return self.smooth_path.get_target(t, pos, dt)
        elif self.cls == "rrt":
            pass

    def reset(self):
        if self.cls == "circle":
            pass
        elif self.cls == "polygon":
            self.current_index = 0
        elif self.cls == "smooth_path":
            self.smooth_path.reset()
        elif self.cls == "rrt":
            pass


if __name__ == "__main__":
    # 运行演示
    demo_smooth_paths()