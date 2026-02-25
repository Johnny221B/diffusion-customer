# import pandas as pd
# import matplotlib.pyplot as plt

# # 1) 读取 CSV
# csv_path = "/home/linyuliu/jxmount/diffusion_custom/outputs/conquest_v21_128d_0223_2144/metrics.csv"
# df = pd.read_csv(csv_path)

# # 2) epoch 转数值并排序
# df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
# df = df.dropna(subset=["epoch"]).sort_values("epoch")

# # 3) 找出数值列（排除 epoch）
# y_cols = [c for c in df.columns if c != "epoch" and pd.api.types.is_numeric_dtype(df[c])]

# # 如果你想固定顺序/固定四列，用下面这一行替换上面 y_cols：
# # y_cols = ["share", "cos_sim", "mse", "norm_mse"]

# # 只取前 4 个（适配 2x2）
# y_cols = y_cols[:4]
# if len(y_cols) < 4:
#     raise ValueError(f"Need at least 4 numeric columns besides 'epoch', got {len(y_cols)}: {y_cols}")

# # 4) 2x2 grid
# fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
# axes = axes.flatten()

# for ax, col in zip(axes, y_cols):
#     ax.plot(df["epoch"], df[col])
#     ax.set_title(f"{col} vs epoch")
#     ax.set_xlabel("epoch")
#     ax.set_ylabel(col)

# fig.tight_layout()
# fig.savefig("metrics_stable.pdf")
# plt.show()

# print("Saved:", "metrics_2x2.png")

import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/home/linyuliu/jxmount/diffusion_custom/outputs/conquest_v21_128d_0223_2144/metrics.csv"
df = pd.read_csv(csv_path)

df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
df = df.dropna(subset=["epoch"]).sort_values("epoch").reset_index(drop=True)

# 你每条记录是“每10个epoch一次”，所以 window=50 表示平滑 50条记录 = 500个epoch
window = 1

for col in ["share", "oracle_share", "regret"]:
    df[col + "_smooth"] = df[col].rolling(window=window, min_periods=1).mean()

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
(ax1, ax2, ax3, ax4) = axes.flatten()

# share & oracle_share（原始=淡，平滑=粗）
ax1.plot(df["epoch"], df["share"], alpha=0.25, label="share (raw)")
ax1.plot(df["epoch"], df["share_smooth"], linewidth=2.0, label=f"share (roll{window})")
ax1.plot(df["epoch"], df["oracle_share"], alpha=0.25, label="oracle_share (raw)")
ax1.plot(df["epoch"], df["oracle_share_smooth"], linewidth=2.0, label=f"oracle_share (roll{window})")
ax1.set_title("share & oracle_share vs epoch")
ax1.set_xlabel("epoch"); ax1.set_ylabel("value")
ax1.legend()

# cos_sim（不平滑也行；如果你也想平滑，同样 rolling 一下即可）
ax2.plot(df["epoch"], df["cos_sim"])
ax2.set_title("cos_sim vs epoch")
ax2.set_xlabel("epoch"); ax2.set_ylabel("cos_sim")

# regret（原始+平滑）
ax3.plot(df["epoch"], df["regret"], alpha=0.25, label="regret (raw)")
ax3.plot(df["epoch"], df["regret_smooth"], linewidth=2.0, label=f"regret (roll{window})")
ax3.set_title("regret vs epoch")
ax3.set_xlabel("epoch"); ax3.set_ylabel("regret")
ax3.legend()

# norm_mse
ax4.plot(df["epoch"], df["norm_mse"])
ax4.set_title("norm_mse vs epoch")
ax4.set_xlabel("epoch"); ax4.set_ylabel("norm_mse")

fig.tight_layout()
# fig.savefig("metrics_R5_better.pdf")
fig.savefig("metrics_stable.pdf")

# fig.savefig("metrics_2x2_smoothed.png", dpi=200)
plt.show()

print("Saved: metrics_2x2_smoothed.pdf / .png")