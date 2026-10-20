import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# ============================================================
# Step 1：模擬板材上的瑕疵座標
# ============================================================

# 兩條彎曲的瑕疵分布，代表兩道刮痕或瑕疵帶
X_defects, _ = make_moons(
    n_samples=700,
    noise=0.04,
    random_state=42
)

# 加入零散的影像誤判點或孤立瑕疵
np.random.seed(42)

X_scattered = np.random.uniform(
    low=[-1.5, -0.8],
    high=[2.5, 1.8],
    size=(40, 2)
)

# 合併座標
X = np.vstack([X_defects, X_scattered])


# ============================================================
# Step 2：使用 DBSCAN 分群
# ============================================================

dbscan = DBSCAN(
    eps=0.15,
    min_samples=8
)

labels = dbscan.fit_predict(X)


# ============================================================
# Step 3：統計分群結果
# ============================================================

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)

print(f"找到的瑕疵群集數：{n_clusters}")
print(f"被標記為雜訊的點數：{n_noise}")


# ============================================================
# Step 4：畫出瑕疵位置與分群結果
# ============================================================

plt.figure(figsize=(8, 6))

for label in sorted(set(labels)):
    mask = labels == label

    if label == -1:
        # DBSCAN 的 -1 代表未歸入任何群集的雜訊點
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            color="black",
            marker="x",
            label="Noise"
        )
    else:
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            alpha=0.7,
            label=f"Defect Cluster {label}"
        )

plt.xlabel("X position on the sheet")
plt.ylabel("Y position on the sheet")
plt.title("DBSCAN: Defect Clustering on a Sheet")
plt.legend()
plt.grid(True)
plt.show()