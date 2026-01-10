import numpy as np

class ThompsonOptimizer:
    def __init__(self, dim_s, dim_z, prior_var=1.0):
        self.dim_s = dim_s
        self.dim_z = dim_z
        # 总维度 d = d_s + d_z + 1 (价格项)
        self.d = dim_s + dim_z + 1 
        
        # 混合效用模型: U = beta*s + gamma*z - alpha*p
        self.mu = np.zeros(self.d)
        self.sigma = np.eye(self.d) * prior_var
        
        self.X = []
        self.y = []

    def sample_theta(self):
        """Phase 0: 汤普森采样，从后验中抽取偏好参数"""
        return np.random.multivariate_normal(self.mu, self.sigma)

    def select_best_x(self, theta, candidates_s, candidates_z, candidates_p):
        """
        根据抽取的 theta，在候选空间中寻找效用最大的产品
        x = (s, z, -p)
        """
        best_val = -float('inf')
        best_idx = 0
        
        for i in range(len(candidates_s)):
            # 构造特征向量，注意价格项取负值，确保其系数 alpha 对应负向效用
            x = np.concatenate([candidates_s[i], candidates_z[i], [-candidates_p[i]]])
            val = np.dot(theta, x)
            if val > best_val:
                best_val = val
                best_idx = i
        return best_idx

    def update(self, x, outcome):
        """Phase 4: 贝叶斯更新后验分布"""
        self.X.append(x)
        self.y.append(outcome)
        
        if len(self.X) < 2: return
        
        X_mat = np.array(self.X)
        y_vec = np.array(self.y)
        
        # 贝叶斯线性回归更新 (在线更新示例)
        precision_prior = np.eye(self.d)
        precision_post = precision_prior + X_mat.T @ X_mat
        self.sigma = np.linalg.inv(precision_post)
        self.mu = self.sigma @ (X_mat.T @ y_vec)