import numpy as np
from scipy.optimize import minimize

class OptimizerVerifier:
    def __init__(self, dim_latent=128):
        self.dim_latent = dim_latent

    def solve_analytical_best(self, theta_v, R, S_matrix):
        """这是你提供的解析解代码"""
        norm_theta = np.linalg.norm(theta_v)
        z_raw = R * (theta_v / norm_theta) if norm_theta > 1e-9 else np.zeros(self.dim_latent)

        if S_matrix is not None and S_matrix.shape[1] > 0:
            S = S_matrix
            # 计算 Gram 矩阵并求逆 (投影操作)
            gram = S.T @ S
            gram_inv = np.linalg.inv(gram + 1e-6 * np.eye(gram.shape[0]))
            coeffs = (z_raw @ S) @ gram_inv
            z_proj = S @ coeffs
            z_perp = z_raw - z_proj
            
            # 重新拉回到半径 R
            if np.linalg.norm(z_perp) > 1e-9:
                z_perp = R * (z_perp / np.linalg.norm(z_perp))
            return z_perp.astype(np.float32)
        return z_raw.astype(np.float32)

def run_verification(dim=128, tokens=32, R=3.0):
    verifier = OptimizerVerifier(dim_latent=dim)
    
    # 随机生成测试数据
    theta_v = np.random.randn(dim).astype(np.float32)
    S_matrix = np.random.randn(dim, tokens).astype(np.float32)

    # 1. 计算解析解
    z_analytical = verifier.solve_analytical_best(theta_v, R, S_matrix)
    utility_analytical = np.dot(theta_v, z_analytical)

    # 2. 通过数值优化寻找最优解 (作为对比标准)
    # 目标函数：最小化 -theta_v^T * z (等价于最大化 Utility)
    def objective(z):
        return -np.dot(theta_v, z)

    # 约束条件
    cons = [
        {'type': 'ineq', 'fun': lambda z: R - np.linalg.norm(z)} # ||z|| <= R
    ]
    # 正交约束：S^T * z = 0 (对每个 token 向量都要正交)
    for i in range(tokens):
        def ortho_constraint(z, idx=i):
            return np.dot(S_matrix[:, idx], z)
        cons.append({'type': 'eq', 'fun': ortho_constraint})

    # 使用 SLSQP 算法求解
    res = minimize(objective, np.zeros(dim), constraints=cons, method='SLSQP', tol=1e-9)
    z_numerical = res.x
    utility_numerical = -res.fun

    # 3. 对比分析
    # 计算两个向量的余弦相似度，看方向是否一致
    cos_sim = np.dot(z_analytical, z_numerical) / (np.linalg.norm(z_analytical) * np.linalg.norm(z_numerical))
    # 检查正交性误差
    ortho_error = np.mean(np.abs(S_matrix.T @ z_analytical))

    print(f"=== 验证结果 (维度: {dim}, R: {R}, Tokens: {tokens}) ===")
    print(f"解析解 Utility (你的方法): {utility_analytical:.8f}")
    print(f"数值解 Utility (最优标准): {utility_numerical:.8f}")
    print(f"解析解模长: {np.linalg.norm(z_analytical):.8f}")
    print(f"数值解模长: {np.linalg.norm(z_numerical):.8f}")
    print(f"余弦相似度 (方向一致性): {cos_sim:.8f}")
    print(f"正交性误差 (mean |S^T z|): {ortho_error:.2e}")

    # 判断是否基本一致
    if np.isclose(utility_analytical, utility_numerical, rtol=1e-4):
        print("\n结论：验证通过！解析解与数值最优解一致。")
    else:
        print("\n结论：存在显著偏差，请检查数学逻辑。")

if __name__ == "__main__":
    run_verification()