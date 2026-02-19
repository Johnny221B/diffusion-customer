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
import time
import numpy as np
from scipy.optimize import minimize

class LogisticThompsonOptimizer:
    def __init__(self, dim_latent=128, prior_var=3.0, exploration_a=1.0): #算utility的时候是在低纬度还是高维度算
        self.dim_latent = dim_latent
        self.d = dim_latent + 1  # 1(Intercept) + 128(Feature)
        self.mu_map = np.zeros(self.d) 
        self.prior_var = prior_var
        self.hessian = np.eye(self.d) / self.prior_var 
        self.exploration_a = exploration_a 
        self.history = []
        self.history_prob = []
        self.eigenvalues = np.zeros(self.d)

    def sample_theta(self):
        H_reg = self.hessian + 1e-4 * np.eye(self.d)
        cov = (self.exploration_a)**2 * np.linalg.inv(H_reg) # theta ~ N(theta, a^2 H_t^-1)
        theta_full = np.random.multivariate_normal(self.mu_map, cov)
        return theta_full[0], theta_full[1:]

    def solve_analytical_best(self, theta_v, R, S_matrix):
        norm_theta = np.linalg.norm(theta_v)
        z_raw = R * (theta_v / norm_theta) if norm_theta > 1e-9 else np.zeros(self.dim_latent)

        if S_matrix is not None and S_matrix.shape[1] > 0:
            S = S_matrix
            # P = I - S(S'S)^-1 S'
            gram = S.T @ S
            gram_inv = np.linalg.inv(gram + 1e-6 * np.eye(gram.shape[0]))
            coeffs = (z_raw @ S) @ gram_inv
            z_proj = S @ coeffs
            z_perp = z_raw - z_proj
            
            if np.linalg.norm(z_perp) > 1e-9:
                z_perp = R * (z_perp / np.linalg.norm(z_perp))
            return z_perp.astype(np.float32)
        return z_raw.astype(np.float32)

    def add_comparison_data(self, v_vector, label):
        """确保 label 以数值形式存储"""
        x = np.concatenate(([1.0], v_vector))
        # 如果传入的是 [y]，则提取 y
        clean_label = label[0] if isinstance(label, (list, np.ndarray)) else label
        self.history.append((x, float(clean_label)))
        
    def update_posterior(self):
        if not self.history: return
        # t_start = time.time()
    
        # 1. 预处理：一次性将 history 转换为矩阵，避免在 neg_log_likelihood 内部循环
        X = np.array([item[0] for item in self.history]) # (N, 128)
        Y = np.array([item[1][0] if isinstance(item[1], (list, np.ndarray)) else item[1] for item in self.history]) # (N,)
        
        # 2. 向量化的损失函数
        def neg_log_likelihood(theta):
            # 正则项
            loss = 0.5 / self.prior_var * np.sum(theta**2)
            # 矩阵乘法代替循环: (N, 128) @ (128,) -> (N,)
            v = X @ theta 
            # 使用 log1p 和 exp 的向量化运算
            loss += np.sum(np.log1p(np.exp(np.clip(v, -20, 20))) - Y * v)
            return loss

        # 3. 向量化的梯度 (提供给 L-BFGS 可以大幅减少其内部尝试次数)
        def gradient(theta):
            v = X @ theta
            prob = 1.0 / (1.0 + np.exp(-np.clip(v, -20, 20)))
            # 梯度公式: X^T @ (prob - y) + theta/sigma^2
            grad = X.T @ (prob - Y) + theta / self.prior_var
            return grad

        # t1 = time.time()
        # 传入 jac=gradient 会让 L-BFGS 跑得飞快
        res = minimize(neg_log_likelihood, self.mu_map, jac=gradient, method='L-BFGS-B') 
        self.mu_map = res.x
        # t_map_solve = time.time() - t1

        # 4. 向量化构建 Hessian
        # t2 = time.time()
        v = X @ self.mu_map
        prob = 1.0 / (1.0 + np.exp(-np.clip(v, -20, 20)))
        weights = prob * (1.0 - prob)
        # 使用矩阵乘法构建: X.T @ diag(weights) @ X
        self.hessian = (X.T * weights) @ X + np.eye(self.d) / self.prior_var
        # t_hessian_build = time.time() - t2

        # t3 = time.time()
        self.eigenvalues = np.linalg.eigvalsh(self.hessian)
        # t_eigen_calc = time.time() - t3
        
        # print(f"\n[Optimized Profile] Epoch Total: {time.time()-t_start:.4f}s")
        # print(f"  - L-BFGS (Vectorized): {t_map_solve:.4f}s")
        # print(f"  - Hessian (Matrix): {t_hessian_build:.4f}s")

    # def update_posterior(self):
    #     if not self.history: return
    #     t_start = time.time()
    
    #     # 1. 向量化预处理 (将 list 转换为 numpy 矩阵)
    #     t0 = time.time()
    #     def neg_log_likelihood(theta):
    #         loss = 0.5/self.prior_var * np.sum(theta**2) # 正则项，这里是否需要一个系数
    #         for x, y in self.history:
    #             # 关键修改：确保 y 是标量数值而非列表
    #             # 如果 y 是 [1]，这里取 y[0]；如果是 1，则保持不变
    #             label = y[0] if isinstance(y, (list, np.ndarray)) else y
                
    #             v = np.dot(x, theta)
    #             # 根据逻辑公式计算负对数似然 objective function
    #             loss += (np.log(1 + np.exp(np.clip(v, -20, 20))) - label * v)
    #         return loss
    #     res = minimize(neg_log_likelihood, self.mu_map, method='L-BFGS-B') #求解theta
    #     self.mu_map = res.x
    #     self.hessian = np.eye(self.d)
    #     # hessian 代表了当前知识的精度（Precision）样本越多、特征方向越集中，在该方向上的 Hessian 特征值就越大，代表模型对该方向的偏好越确定 。
    #     for x, _ in self.history:
    #         v = np.dot(x, self.mu_map)
    #         prob = 1.0 / (1.0 + np.exp(-np.clip(v, -20, 20))) # sigmoid 概率计算
    #         self.history_prob.append(prob)
    #         weight = prob * (1.0 - prob) # 导数的结果
    #         self.hessian += weight * np.outer(x, x)
    #     self.eigenvalues = np.linalg.eigvalsh(self.hessian)
        # print(self.history_prob)
        # print(f"the max is {np.max(self.eigenvalues):.4f}")
        # print(f"the min is {np.min(self.eigenvalues):.4f}")
        
            
            
            
# class LogisticThompsonOptimizer:
#     def __init__(self, dim_latent=128, prior_var=1.0, exploration_a=1.0):
#         self.dim_latent = dim_latent 
#         self.d = dim_latent 
#         self.mu_map = np.zeros(self.d) 
#         self.hessian = np.eye(self.d) / prior_var 
#         self.exploration_a = exploration_a 
#         self.history = [] 

#     def sample_theta(self): 
#         H_reg = self.hessian + 1e-4 * np.eye(self.d) 
#         cov = (self.exploration_a)**2 * np.linalg.inv(H_reg) 
#         return np.random.multivariate_normal(self.mu_map, cov) 

#     def solve_analytical_best(self, theta, R, S_matrix):
#             """
#             矩阵版垂直操作：将 z 投影到由 S_matrix 中所有 Token 组成的子空间的正交补空间。
            
#             参数:
#             - theta: 采样得到的偏好向量 (128,)
#             - R: 固定探索半径
#             - S_matrix: 投影后的 Prompt 矩阵 (128, n_tokens)
#             """
#             # 1. 基础解
#             norm_theta = np.linalg.norm(theta)
#             z_raw = R * (theta / norm_theta) if norm_theta > 1e-9 else np.zeros(self.dim_latent)

#             # 2. 矩阵版正交投影
#             # S 形状为 (128, n)，我们计算 P = I - S(S'S)^-1 S'
#             if S_matrix is not None and S_matrix.shape[1] > 0:
#                 # 这里的 S 就是 S_matrix
#                 S = S_matrix
                
#                 # 计算 Gram 矩阵 S'S 并求逆 (加个微小扰动防止奇异)
#                 gram = S.T @ S
#                 gram_inv = np.linalg.inv(gram + 1e-6 * np.eye(gram.shape[0]))
                
#                 # 计算投影分量: z_proj = S @ (S'S)^-1 @ S' @ z_raw
#                 # 我们直接用 z_raw @ S 来加速计算
#                 coeffs = (z_raw @ S) @ gram_inv
#                 z_proj = S @ coeffs
                
#                 # 垂直分量
#                 z_perp = z_raw - z_proj
                
#                 # 重新拉回到半径 R，保证能量一致
#                 if np.linalg.norm(z_perp) > 1e-9:
#                     z_perp = R * (z_perp / np.linalg.norm(z_perp))
#                 return z_perp.astype(np.float32)

#             return z_raw.astype(np.float32)

#     def add_comparison_data(self, delta_x, labels):
#         for y in labels:
#             self.history.append((delta_x, y))

#     def update_posterior(self):
#         if not self.history: return
#         def neg_log_likelihood(theta):
#             loss = 0.5 * np.sum(theta**2) 
#             for dx, y in self.history:
#                 v = np.dot(dx, theta)
#                 loss += (np.log(1 + np.exp(np.clip(v, -20, 20))) - y * v)
#             return loss
#         res = minimize(neg_log_likelihood, self.mu_map, method='L-BFGS-B')
#         self.mu_map = res.x
#         self.hessian = np.eye(self.d)
#         for dx, _ in self.history:
#             v = np.dot(dx, self.mu_map)
#             prob = 1.0 / (1.0 + np.exp(-np.clip(v, -20, 20)))
#             weight = prob * (1.0 - prob)
#             self.hessian += weight * np.outer(dx, dx)

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