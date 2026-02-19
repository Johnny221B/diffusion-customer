import pandas as pd
import matplotlib.pyplot as plt

# 1) 读取 CSV
csv_path = "/home/linyuliu/jxmount/diffusion_custom/outputs/conquest_v18_128d_0217_1438/metrics.csv"
df = pd.read_csv(csv_path)

# 2) epoch 转数值并排序
df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
df = df.dropna(subset=["epoch"]).sort_values("epoch")

# 3) 找出数值列（排除 epoch）
y_cols = [c for c in df.columns if c != "epoch" and pd.api.types.is_numeric_dtype(df[c])]

# 如果你想固定顺序/固定四列，用下面这一行替换上面 y_cols：
# y_cols = ["share", "cos_sim", "mse", "norm_mse"]

# 只取前 4 个（适配 2x2）
y_cols = y_cols[:4]
if len(y_cols) < 4:
    raise ValueError(f"Need at least 4 numeric columns besides 'epoch', got {len(y_cols)}: {y_cols}")

# 4) 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten()

for ax, col in zip(axes, y_cols):
    ax.plot(df["epoch"], df[col])
    ax.set_title(f"{col} vs epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel(col)

fig.tight_layout()
fig.savefig("metrics_2x2.pdf")
plt.show()

print("Saved:", "metrics_2x2.png")
