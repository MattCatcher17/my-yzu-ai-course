# Step 0：載入套件
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
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
    + np.random.normal(0, temperature_noise * noise_coefficient, n)
)

data["vibration"] = (
    data["true_vibration"]
    + np.random.normal(0, vibration_noise * noise_coefficient, n)
)

data["rpm"] = (
    data["true_rpm"]
    + np.random.normal(0, rpm_noise * noise_coefficient, n)
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

print("Input Data:")
display(X.head(10))


# ============================================================
# Step 4：Standardization
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# Step 5：建立 DBSCAN 模型
# ============================================================

dbscan = DBSCAN(
    eps=0.8,          # 鄰近距離：可嘗試調整
    min_samples=10    # 成為核心點所需的最少鄰近資料點數
)

# DBSCAN 使用 fit_predict 同時訓練並取得分群結果
data["cluster"] = dbscan.fit_predict(X_scaled)


# ============================================================
# Step 6：查看 DBSCAN 分群結果
# ============================================================

display(
    data[
        [
            "temperature",
            "vibration",
            "rpm",
            "cluster"
        ]
    ].head(20)
)

# DBSCAN 將 -1 表示為雜訊點
noise_count = (data["cluster"] == -1).sum()

# 計算群數時排除 -1
n_clusters = len(set(data["cluster"])) - (1 if -1 in data["cluster"].values else 0)

print(f"DBSCAN 找到的群數：{n_clusters}")
print(f"被標記為雜訊的資料筆數：{noise_count}")


# ============================================================
# Step 7：觀察每一群的特徵
# ============================================================

cluster_summary = data.groupby("cluster")[
    [
        "temperature",
        "vibration",
        "rpm"
    ]
].mean()

print("Cluster Summary:")
display(cluster_summary)

# cluster = -1 是雜訊點的平均特徵，可和其他群比較


# ============================================================
# Step 8：使用 PCA 將資料降到 2D
# ============================================================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

data["PC1"] = X_pca[:, 0]
data["PC2"] = X_pca[:, 1]


# ============================================================
# Step 9：畫出 DBSCAN 分群結果
# ============================================================

plt.figure(figsize=(8, 6))

cluster_labels = sorted(data["cluster"].unique())
colors = plt.cm.tab10(np.linspace(0, 1, max(len(cluster_labels), 1)))

for color, label in zip(colors, cluster_labels):
    cluster_data = data[data["cluster"] == label]

    if label == -1:
        # 雜訊點用黑色叉號表示
        plt.scatter(
            cluster_data["PC1"],
            cluster_data["PC2"],
            color="black",
            marker="x",
            alpha=0.7,
            label="Noise (-1)"
        )
    else:
        plt.scatter(
            cluster_data["PC1"],
            cluster_data["PC2"],
            color=color,
            alpha=0.6,
            label=f"Cluster {label}"
        )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Machine Operating Data - DBSCAN Clustering")
plt.legend()
plt.show()