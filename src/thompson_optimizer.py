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
        beta = theta[:self.dim_s]       # 结构化属性偏好
        gamma = theta[self.dim_s : self.dim_s + self.dim_z] # 风格偏好
        alpha = theta[-1]               # 价格敏感度 (通常 alpha > 0)

        if np.linalg.norm(gamma) > 1e-9:
            best_z = R * (gamma / np.linalg.norm(gamma))
        else:
            best_z = np.zeros(self.dim_z)

        best_s = np.where(beta > 0, 1, 0)

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