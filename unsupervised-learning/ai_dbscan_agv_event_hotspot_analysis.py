# ============================================================
# DBSCAN：AGV 近失事件熱點分析
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN


# ============================================================
# Step 1：模擬工廠內的 AGV 事件位置
# ============================================================

rng = np.random.default_rng(42)

# 模擬三個可能的事件熱點，座標單位為公尺
hotspots = {
    "中央路口": {"center": (20, 15), "count": 80},
    "加工站入口": {"center": (8, 6), "count": 70},
    "狹窄通道": {"center": (33, 7), "count": 70}
}

event_types = ["急停", "人工介入避讓", "近失事件"]
records = []

# 在每個熱點周圍模擬事件位置
for location_name, settings in hotspots.items():
    center_x, center_y = settings["center"]
    count = settings["count"]

    x_positions = rng.normal(center_x, 1.2, count)
    y_positions = rng.normal(center_y, 1.2, count)

    for x, y in zip(x_positions, y_positions):
        records.append({
            "x_m": x,
            "y_m": y,
            "event_type": rng.choice(event_types),
            "simulated_area": location_name
        })

# 加入零散事件，模擬其他位置偶發的急停或人工介入
n_scattered_events = 30

for _ in range(n_scattered_events):
    records.append({
        "x_m": rng.uniform(0, 40),
        "y_m": rng.uniform(0, 25),
        "event_type": rng.choice(event_types),
        "simulated_area": "零散事件"
    })

data = pd.DataFrame(records)

print("事件資料預覽：")
display(data.head())


# ============================================================
# Step 2：準備 DBSCAN 輸入資料
# ============================================================

# 只用事件座標分群，不使用事件類型或模擬區域名稱
X = data[["x_m", "y_m"]]


# ============================================================
# Step 3：執行 DBSCAN
# ============================================================

dbscan = DBSCAN(
    eps=2.5,         # 2.5 公尺內視為鄰近
    min_samples=8    # 至少 8 個鄰近事件才形成密集區域
)

data["cluster"] = dbscan.fit_predict(X)


# ============================================================
# Step 4：查看群集數量與事件筆數
# ============================================================

n_clusters = len(set(data["cluster"])) - (
    1 if -1 in data["cluster"].values else 0
)

n_noise = (data["cluster"] == -1).sum()

print(f"找到的事件熱點數：{n_clusters}")
print(f"未形成密集群集的事件數：{n_noise}")
print("\n各群事件筆數：")
print(data["cluster"].value_counts().sort_index())


# 顯示各群的座標平均值，協助定位熱點
cluster_summary = (
    data[data["cluster"] != -1]
    .groupby("cluster")[["x_m", "y_m"]]
    .mean()
    .rename(columns={"x_m": "平均 X 座標（m）", "y_m": "平均 Y 座標（m）"})
)

print("\n各熱點的平均位置：")
display(cluster_summary)


# ============================================================
# Step 5：繪製事件位置與 DBSCAN 分群結果
# ============================================================

plt.figure(figsize=(10, 7))

for cluster_id in sorted(data["cluster"].unique()):
    subset = data[data["cluster"] == cluster_id]

    if cluster_id == -1:
        plt.scatter(
            subset["x_m"],
            subset["y_m"],
            color="black",
            marker="x",
            s=60,
            label="Scattered events (-1)"
        )
    else:
        plt.scatter(
            subset["x_m"],
            subset["y_m"],
            s=45,
            alpha=0.7,
            label=f"Hotspot {cluster_id}"
        )

# 以方形標示模擬的路口、站點與狹窄通道位置
for name, settings in hotspots.items():
    x, y = settings["center"]
    plt.scatter(x, y, marker="s", s=150, facecolors="none", edgecolors="red")
    plt.text(x + 0.5, y + 0.5, name, color="red")

plt.xlim(0, 40)
plt.ylim(0, 25)
plt.xlabel("X position in factory (m)")
plt.ylabel("Y position in factory (m)")
plt.title("DBSCAN: AGV Near-Miss Event Hotspots")
plt.grid(True)
plt.legend()
plt.show()