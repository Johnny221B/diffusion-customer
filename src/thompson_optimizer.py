import numpy as np

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