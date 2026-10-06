# ============================================================
# AI 人工智慧概論：非監督式學習(Unsupervised Learning)
# K-Means Clustering
#
# 課程目標：
# 不提供 Normal / Warning / Danger 標籤
# 讓 AI 自己從機器運轉資料中找出不同的群組
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


print("Input Data:")

display(X.head(10))


# ============================================================
# Step 4：Standardization
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# Step 5：建立 K-Means 模型
# ============================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)


# ============================================================
# Step 6：取得 AI 找到的 Cluster
# ============================================================

data["cluster"] = kmeans.labels_


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


# ============================================================
# Step 8：使用 PCA 將資料降到 2D
# ============================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)


data["PC1"] = X_pca[:, 0]
data["PC2"] = X_pca[:, 1]


# ============================================================
# Step 9：畫出 Cluster
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    data["PC1"],
    data["PC2"],
    c=data["cluster"],
    alpha=0.6
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "Machine Operating Data - K-Means Clustering"
)

plt.show()