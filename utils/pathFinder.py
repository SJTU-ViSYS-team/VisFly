import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import kneighbors_graph
import networkx as nx
from magnum import Range3D
class PRMPlanner:
    def __init__(self, bounds, num_samples, obstacle_func, k=10, scene_id=None):
        """
        初始化PRM规划器。
        :param bounds: 空间的边界，形如((min_x, max_x), (min_y, max_y), (min_z, max_z))
        :param num_samples: 随机采样的数量
        :param obstacle_func: 障碍物检测函数，输入为点坐标，输出为布尔值，True表示是障碍物
        :param k: 构建路线图时每个节点的近邻数量
        """
        self._bounds = None
        self.bounds = bounds
        self.num_samples = num_samples
        self.obstacle_func = obstacle_func
        self.k = k
        self.samples = []
        self.graph = nx.Graph()

        self.scene_id = scene_id


    def sample_free_space(self):
        """在自由空间内随机采样点"""
        while len(self.samples) < self.num_samples:
            point = np.random.uniform([b[0] for b in self._bounds], [b[1] for b in self._bounds])
            if not self.obstacle_func(point, self.scene_id):
                self.samples.append(point)
        self.samples = np.array(self.samples)

    def pre_plan(self):
        self.sample_free_space()
        self.build_roadmap()

    def build_roadmap(self):
        """构建路线图"""
        # 使用KDTree快速查找近邻
        tree = KDTree(self.samples)
        # 构建近邻图
        A = kneighbors_graph(self.samples, self.k, mode='distance', metric='euclidean', include_self=False)
        A = A.toarray()
        
        # 添加边到图中
        for i, neighbors in enumerate(A):
            for j, dist in enumerate(neighbors):
                if dist > 0 and not self.obstacle_func((self.samples[i] + self.samples[j]) / 2, self.scene_id):  # 确保路径中间没有障碍物
                    self.graph.add_edge(i, j, weight=dist)

    def plan(self, start, goal):
        """规划从起点到终点的路径"""
        # 将起点和终点加入图中
        start_idx = len(self.samples)
        goal_idx = start_idx + 1
        self.samples = np.vstack([self.samples, start, goal])
        self.graph.add_node(start_idx)
        self.graph.add_node(goal_idx)
        
        # 将起点和终点连接到图中
        tree = KDTree(self.samples)
        start_neighbors = tree.query(start, self.k)[1]
        goal_neighbors = tree.query(goal, self.k)[1]
        
        for neighbor in start_neighbors:
            if not self.obstacle_func((start + self.samples[neighbor]) / 2, self.scene_id):
                self.graph.add_edge(start_idx, neighbor, weight=np.linalg.norm(start - self.samples[neighbor]))
        for neighbor in goal_neighbors:
            if not self.obstacle_func((goal + self.samples[neighbor]) / 2, self.scene_id):
                self.graph.add_edge(goal_idx, neighbor, weight=np.linalg.norm(goal - self.samples[neighbor]))
        
        # 使用A*算法查找最短路径
        path = nx.astar_path(self.graph, start_idx, goal_idx, heuristic=lambda a, b: np.linalg.norm(self.samples[a] - self.samples[b]))
        return [self.samples[i] for i in path]

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        if isinstance(bounds, Range3D):
            self._bounds = np.array([bounds.min, bounds.max]).T
        else:
            self._bounds = bounds

def debug():
    def draw_ball(ax, data):
        center = data[:3]
        radius = data[3]
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        color = cm.coolwarm(z/7)
        ax.plot_surface(x, y, z, facecolors=color, alpha=0.5)

        
    bounds = np.array([[0,0,0],[10,5,5]])
    ball_num = 20
    
    def ball_generate(num):
        pos = np.random.uniform(bounds[0], bounds[1], [num, 3])
        r = np.random.uniform(0.5,1,num)
        return np.c_[pos, r]
    balls = ball_generate(ball_num)
    
    # 示例：定义障碍物检测函数
    def is_obstacle(point, scene_id=None):
        is_ob = False
        # 假设有一个球形障碍物在空间中心
        for data in balls:
            center = data[:3]
            radius = data[3]
            if np.linalg.norm(point - center) < radius:
                is_ob = True
                break
            
        return is_ob 

    import time
    
    # 创建PRM规划器实例
    start = time.time()
    planner = PRMPlanner(bounds=((0, 10), (0, 10), (0, 10)), num_samples=500, obstacle_func=is_obstacle, k=10,scene_id=0)
    planner.pre_plan()
    # planner.sample_free_space()
    # planner.build_roadmap()
    end = time.time()
    print(f"time:{end-start}%5f")
    # 规划路径
    start = np.array([1, 1, 1])
    goal = np.array([9, 3, 3])
    path = planner.plan(start, goal)

    print("Path from start to goal:", path)

    from matplotlib import pyplot as plt
    from matplotlib import cm
    # plot 3d path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for data in balls:
        draw_ball(ax, data)
    # plot 3d path
    path = np.array(path)  # convert path to numpy array for easier slicing

    ax.plot(path[:,0], path[:,1], path[:,2], linewidth=3, color="red")
    ax.set_aspect("equal")
    plt.show()
    fig.savefig('debug/pathfinder_demo.png')
    
if __name__ == "__main__":
    debug()