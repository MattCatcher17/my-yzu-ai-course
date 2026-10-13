# ============================================================
# AI 人工智慧概論：非監督式學習(Unsupervised Learning)
# K-Means Clustering 範例2

#
# 課程目標：比較有無 Standardization 的 K-Means 分群結果
# ============================================================


# ============================================================
# Step 0：載入套件
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# Step 1：產生模擬的機器運轉資料
# ============================================================

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "true_temperature": np.random.normal(75, 10, n),
    "true_vibration": np.random.normal(4, 1.5, n),
    "true_rpm": np.random.normal(1800, 300, n)
})


# 限制合理範圍
data["true_temperature"] = data["true_temperature"].clip(40, 110)
data["true_vibration"] = data["true_vibration"].clip(0.5, 10)
data["true_rpm"] = data["true_rpm"].clip(800, 3000)


# ============================================================
# Step 2：加入 Sensor Noise
# ============================================================

temperature_noise = 2.0
vibration_noise = 0.5
rpm_noise = 80

noise_coefficient = 1.0


data["temperature"] = (
    data["true_temperature"]
    + np.random.normal(
        0,
        temperature_noise * noise_coefficient,
        n
    )
)


data["vibration"] = (
    data["true_vibration"]
    + np.random.normal(
        0,
        vibration_noise * noise_coefficient,
        n
    )
)


data["rpm"] = (
    data["true_rpm"]
    + np.random.normal(
        0,
        rpm_noise * noise_coefficient,
        n
    )
)


# ============================================================
# Step 3：準備 AI 的輸入 X
# ============================================================

X = data[
    [
        "temperature",
        "vibration",
        "rpm"
    ]
]


# ============================================================
# Step 4：Standardization
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# Step 5：建立兩個 K-Means 模型
# ============================================================

# Model A：沒有標準化
model_A = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

model_A.fit(X)

# Model B：有標準化
model_B = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

model_B.fit(X_scaled)


# ============================================================
# Step 6：儲存分群結果
# ============================================================

data["cluster_A"] = model_A.labels_
data["cluster_B"] = model_B.labels_


# ============================================================
# Step 7：觀察每一群的特徵
# ============================================================

print("Model A - Without Standardization")
display(
    data.groupby("cluster_A")[
        ["temperature", "vibration", "rpm"]
    ].mean()
)

print("Model B - With Standardization")
display(
    data.groupby("cluster_B")[
        ["temperature", "vibration", "rpm"]
    ].mean()
)


# ============================================================
# Step 8：PCA 降維
# ============================================================

# 使用同一組標準化後的資料建立 PCA
# 讓兩個模型使用相同的 2D 座標系統

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)


# ============================================================
# Step 9：畫出兩個 K-Means 分群結果
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Model A：沒有標準化
axes[0].scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=data["cluster_A"],
    cmap="viridis",
    alpha=0.6,
    s=15
)

axes[0].set_title(
    "Model A - Without Standardization"
)

axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")


# Model B：有標準化
axes[1].scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=data["cluster_B"],
    cmap="viridis",
    alpha=0.6,
    s=15
)

axes[1].set_title(
    "Model B - With Standardization"
)

axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")

plt.tight_layout()
plt.show()


# ============================================================
# Step 10：3D K-Means Clustering
# 比較有無 Standardization 的分群結果
# ============================================================

fig = plt.figure(figsize=(14, 6))

# Model A：Without Standardization
ax1 = fig.add_subplot(121, projection="3d")

ax1.scatter(
    data["temperature"],
    data["vibration"],
    data["rpm"],
    c=data["cluster_A"],
    cmap="viridis",
    s=12,
    alpha=0.6
)

ax1.set_title("Model A - Without Standardization")
ax1.set_xlabel("Temperature (°C)")
ax1.set_ylabel("Vibration")
ax1.set_zlabel("RPM")


# Model B：With Standardization
ax2 = fig.add_subplot(122, projection="3d")

ax2.scatter(
    data["temperature"],
    data["vibration"],
    data["rpm"],
    c=data["cluster_B"],
    cmap="viridis",
    s=12,
    alpha=0.6
)

ax2.set_title("Model B - With Standardization")
ax2.set_xlabel("Temperature (°C)")
ax2.set_ylabel("Vibration")
ax2.set_zlabel("RPM")


# 設定相同視角與座標範圍，方便比較
for ax in [ax1, ax2]:
    ax.view_init(elev=22, azim=45)
    ax.set_xlim(X["temperature"].min(), X["temperature"].max())
    ax.set_ylim(X["vibration"].min(), X["vibration"].max())
    ax.set_zlim(X["rpm"].min(), X["rpm"].max())

plt.tight_layout()
plt.show()