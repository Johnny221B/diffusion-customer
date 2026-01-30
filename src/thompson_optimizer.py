import numpy as np
from scipy.optimize import minimize

class ThompsonOptimizer:
    def __init__(self, dim_latent, prior_var=1.0):
        # 这里的参数名必须是 dim_latent
        self.dim_latent = dim_latent
        self.d = dim_latent + 1 
        self.mu = np.zeros(self.d)
        self.sigma = np.eye(self.d) * prior_var
        self.X_buffer = []
        self.y_buffer = []

    def sample_theta(self):
        return np.random.multivariate_normal(self.mu, self.sigma)

    def solve_analytical_best(self, theta, R, price_range=(50, 200)):
        gamma = theta[:self.dim_latent]
        alpha = theta[-1]
        if np.linalg.norm(gamma) > 1e-9:
            best_z_latent = R * (gamma / np.linalg.norm(gamma))
        else:
            best_z_latent = np.zeros(self.dim_latent)
        best_p = price_range[0] if alpha > 0 else price_range[1]
        return best_z_latent.astype(np.float32), best_p

    def add_to_buffer(self, x, outcome):
        self.X_buffer.append(x)
        self.y_buffer.append(outcome)

    def update_from_buffer(self):
        if not self.X_buffer: return
        X_mat = np.array(self.X_buffer)
        y_vec = np.array(self.y_buffer)
        precision_post = np.eye(self.d) + X_mat.T @ X_mat
        self.sigma = np.linalg.inv(precision_post)
        self.mu = self.sigma @ (X_mat.T @ y_vec)
        self.X_buffer, self.y_buffer = [], []

    def get_max_eigenvalue(self):
        return np.linalg.norm(self.sigma, ord=2)

class LogisticThompsonOptimizer:
    def __init__(self, dim_latent, prior_var=1.0, exploration_a=1.0):
        self.dim_latent = dim_latent
        self.d = dim_latent + 1 
        self.mu_map = np.zeros(self.d) 
        self.hessian = np.eye(self.d) / prior_var 
        self.exploration_a = exploration_a 
        self.history = []

    def sample_theta(self):
        # 使用拉普拉斯近似采样 [cite: 78, 192]
        cov = self.exploration_a**2 * np.linalg.inv(self.hessian)
        return np.random.multivariate_normal(self.mu_map, cov)

    def solve_analytical_best(self, theta, R):
        # 基于线性效用 U = x'theta 寻找最优 x [cite: 72]
        gamma = theta[:self.dim_latent]
        if np.linalg.norm(gamma) > 1e-9:
            best_z = R * (gamma / np.linalg.norm(gamma))
        else:
            best_z = np.zeros(self.dim_latent)
        # 价格效用 -alpha*p，权重 alpha>0 时选低价 50
        best_p = 50.0 if theta[-1] > 0 else 200.0
        return best_z.astype(np.float32), best_p

    def add_comparison_data(self, delta_x, labels):
        for y in labels:
            self.history.append((delta_x, y))

    def update_posterior(self):
        if not self.history: return

        # 寻找 MLE [cite: 79]
        def neg_log_likelihood(theta):
            loss = 0.5 * np.sum(theta**2) 
            for dx, y in self.history:
                v = np.dot(dx, theta)
                # Clip 避免数值溢出
                loss += (np.log(1 + np.exp(np.clip(v, -20, 20))) - y * v)
            return loss

        res = minimize(neg_log_likelihood, self.mu_map, method='L-BFGS-B')
        self.mu_map = res.x

        # 更新海森矩阵 Hessian [cite: 80, 52]
        self.hessian = np.eye(self.d)
        for dx, _ in self.history:
            v = np.dot(dx, self.mu_map)
            prob = 1.0 / (1.0 + np.exp(-np.clip(v, -20, 20)))
            weight = prob * (1.0 - prob)
            self.hessian += weight * np.outer(dx, dx)