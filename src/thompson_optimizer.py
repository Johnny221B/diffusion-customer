import numpy as np

class ThompsonOptimizer:
    def __init__(self, dim_s, dim_z, prior_var=1.0):
        self.dim_s = dim_s
        self.dim_z = dim_z
        self.d = dim_s + dim_z + 1 
        self.mu = np.zeros(self.d)
        self.sigma = np.eye(self.d) * prior_var
        self.X, self.y = [], []

    def sample_theta(self):
        """Phase 0: 采样偏好参数 theta"""
        return np.random.multivariate_normal(self.mu, self.sigma)

    def solve_analytical_best(self, theta, R=0.5):
        """
        核心逻辑：计算风格向量 z 的显式解
        Returns: (best_s, best_z, best_p)
        """
        # 1. 提取各部分偏好参数
        beta = theta[:self.dim_s]       # 结构化属性偏好
        gamma = theta[self.dim_s : self.dim_s + self.dim_z] # 风格偏好
        alpha = theta[-1]               # 价格敏感度 (通常 alpha > 0)

        # 2. 计算 z 的显式解：沿着 gamma 方向推到半径 R 的边界
        if np.linalg.norm(gamma) > 1e-9:
            best_z = R * (gamma / np.linalg.norm(gamma))
        else:
            best_z = np.zeros(self.dim_z)

        # 3. 决定属性 s：在线性效用下，取能使 beta * s 最大的离散值 (0 或 1)
        best_s = np.where(beta > 0, 1, 0)

        # 4. 决定价格 p：由于效用包含 -alpha * p，若 alpha > 0，最优价格为最低价
        # 这里我们模拟一个价格区间 [50, 200]
        best_p = 50.0 if alpha > 0 else 200.0

        return best_s, best_z.astype(np.float32), best_p

    def update(self, x, outcome):
        """Phase 4: 贝叶斯更新"""
        self.X.append(x)
        self.y.append(outcome)
        if len(self.X) < 2: return
        X_mat, y_vec = np.array(self.X), np.array(self.y)
        prec_post = np.eye(self.d) + X_mat.T @ X_mat
        self.sigma = np.linalg.inv(prec_post)
        self.mu = self.sigma @ (X_mat.T @ y_vec)