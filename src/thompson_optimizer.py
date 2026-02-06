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


# \src\thompson_optimizer

import numpy as np
from scipy.optimize import minimize

class LogisticThompsonOptimizer:
    def __init__(self, dim_latent=128, prior_var=1.0, exploration_a=1.0):
        self.dim_latent = dim_latent
        self.d = dim_latent 
        self.mu_map = np.zeros(self.d) 
        self.hessian = np.eye(self.d) / prior_var 
        self.exploration_a = exploration_a 
        self.history = []

    def sample_theta(self):
        H_reg = self.hessian + 1e-4 * np.eye(self.d)
        cov = (self.exploration_a)**2 * np.linalg.inv(H_reg)
        return np.random.multivariate_normal(self.mu_map, cov)

    def solve_analytical_best(self, theta, R, S_matrix):
            """
            矩阵版垂直操作：将 z 投影到由 S_matrix 中所有 Token 组成的子空间的正交补空间。
            
            参数:
            - theta: 采样得到的偏好向量 (128,)
            - R: 固定探索半径
            - S_matrix: 投影后的 Prompt 矩阵 (128, n_tokens)
            """
            # 1. 基础解
            norm_theta = np.linalg.norm(theta)
            z_raw = R * (theta / norm_theta) if norm_theta > 1e-9 else np.zeros(self.dim_latent)

            # 2. 矩阵版正交投影
            # S 形状为 (128, n)，我们计算 P = I - S(S'S)^-1 S'
            if S_matrix is not None and S_matrix.shape[1] > 0:
                # 这里的 S 就是 S_matrix
                S = S_matrix
                
                # 计算 Gram 矩阵 S'S 并求逆 (加个微小扰动防止奇异)
                gram = S.T @ S
                gram_inv = np.linalg.inv(gram + 1e-6 * np.eye(gram.shape[0]))
                
                # 计算投影分量: z_proj = S @ (S'S)^-1 @ S' @ z_raw
                # 我们直接用 z_raw @ S 来加速计算
                coeffs = (z_raw @ S) @ gram_inv
                z_proj = S @ coeffs
                
                # 垂直分量
                z_perp = z_raw - z_proj
                
                # 重新拉回到半径 R，保证能量一致
                if np.linalg.norm(z_perp) > 1e-9:
                    z_perp = R * (z_perp / np.linalg.norm(z_perp))
                return z_perp.astype(np.float32)

            return z_raw.astype(np.float32)

    def add_comparison_data(self, delta_x, labels):
        for y in labels:
            self.history.append((delta_x, y))

    def update_posterior(self):
        if not self.history: return
        def neg_log_likelihood(theta):
            loss = 0.5 * np.sum(theta**2) 
            for dx, y in self.history:
                v = np.dot(dx, theta)
                loss += (np.log(1 + np.exp(np.clip(v, -20, 20))) - y * v)
            return loss
        res = minimize(neg_log_likelihood, self.mu_map, method='L-BFGS-B')
        self.mu_map = res.x
        self.hessian = np.eye(self.d)
        for dx, _ in self.history:
            v = np.dot(dx, self.mu_map)
            prob = 1.0 / (1.0 + np.exp(-np.clip(v, -20, 20)))
            weight = prob * (1.0 - prob)
            self.hessian += weight * np.outer(dx, dx)

# import numpy as np
# from scipy.optimize import minimize

# class LogisticThompsonOptimizer:
#     def __init__(self, dim_latent, prior_var=1.0, exploration_a=1.0):
#         self.dim_latent = dim_latent
#         self.d = dim_latent + 1 
#         self.mu_map = np.zeros(self.d) 
#         self.hessian = np.eye(self.d) / prior_var 
#         self.exploration_a = exploration_a 
#         self.history = []

#     def sample_theta(self, cooling_factor=1.0):
#         """
#         Samples from the Laplace approximation.
#         cooling_factor: Scales the exploration variance down as the experiment progresses.
#         """
#         # Add regularization to ensure the matrix is invertible
#         H_reg = self.hessian + 1e-4 * np.eye(self.d)
#         cov = (self.exploration_a * cooling_factor)**2 * np.linalg.inv(H_reg)
#         return np.random.multivariate_normal(self.mu_map, cov)
#         # cooling不需要了

#     def solve_analytical_best(self, theta, R):
#         gamma = theta[:self.dim_latent]
#         if np.linalg.norm(gamma) > 1e-9:
#             best_z = R * (gamma / np.linalg.norm(gamma))
#         else:
#             best_z = np.zeros(self.dim_latent)
#         # Utility U = z*gamma - p*alpha. If alpha (theta[-1]) > 0, pick low price.
#         best_p = 50.0 if theta[-1] > 0 else 200.0
#         return best_z.astype(np.float32), best_p
#     # 先投影先解z有没有区别用例子试一下，price去掉

#     def add_comparison_data(self, delta_x, labels):
#         for y in labels:
#             self.history.append((delta_x, y))

#     def update_posterior(self):
#         if not self.history: return
        
#         def neg_log_likelihood(theta):
#             # L2 regularization corresponds to the Gaussian prior
#             loss = 0.5 * np.sum(theta**2) 
#             for dx, y in self.history:
#                 v = np.dot(dx, theta)
#                 # Logistic NLL = log(1+exp(v)) - y*v
#                 loss += (np.log(1 + np.exp(np.clip(v, -20, 20))) - y * v)
#             return loss

#         res = minimize(neg_log_likelihood, self.mu_map, method='L-BFGS-B')
#         self.mu_map = res.x

#         # Recompute Hessian at the new MAP estimate
#         self.hessian = np.eye(self.d)
#         for dx, _ in self.history:
#             v = np.dot(dx, self.mu_map)
#             prob = 1.0 / (1.0 + np.exp(-np.clip(v, -20, 20)))
#             weight = prob * (1.0 - prob)
#             self.hessian += weight * np.outer(dx, dx)