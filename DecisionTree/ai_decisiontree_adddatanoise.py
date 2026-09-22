
# ============================================================
# AI 人工智慧概論：Data Noise
# ============================================================

# Step 0：載入套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# Step 1：產生模擬的機器運轉資料
# ============================================================

# 固定亂數種子，讓每次執行結果一致
np.random.seed(42)

# 資料筆數
n = 1000

data = pd.DataFrame({
    "true_temperature": np.random.normal(75, 10, n),
    "true_vibration": np.random.normal(4, 1.5, n),
    "true_rpm": np.random.normal(1800, 300, n)
})

data["true_temperature"] = data["true_temperature"].clip(40, 110)
data["true_vibration"] = data["true_vibration"].clip(0.5, 10)
data["true_rpm"] = data["true_rpm"].clip(800, 3000)

print("前 5 筆機器運轉資料：")
display(data.head())

# ============================================================
# Step 2：加入 Sensor Noise
# ============================================================

temperature_noise = 2.0
vibration_noise = 0.5
rpm_noise = 80
noise_coefficient= 1


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

display(
    data[
        [
            "true_temperature",
            "temperature",
            "true_vibration",
            "vibration",
            "true_rpm",
            "rpm"
        ]
    ].head(10)
)

# ============================================================
# Step 3：根據真實狀態建立 Ground Truth
# ============================================================

def engineering_rule(row):

    if (
        row["true_temperature"] > 90
        or row["true_vibration"] > 7
        or row["true_rpm"] > 2500
    ):
        return "Danger"

    elif (
        row["true_temperature"] > 80
        or row["true_vibration"] > 5
        or row["true_rpm"] > 2200
    ):
        return "Warning"

    else:
        return "Normal"


data["true_status"] = data.apply(
    engineering_rule,
    axis=1
)

display(
    data[
        [
            "true_temperature",
            "temperature",
            "true_vibration",
            "vibration",
            "true_rpm",
            "rpm",
            "true_status"
        ]
    ].head(10)
)

X = data[
    [
        "temperature",
        "vibration",
        "rpm"
    ]
]

y = data["true_status"]

print("AI 輸入 X：")
display(X.head())

print("答案 y：")
display(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Training data:", len(X_train))
print("Testing data:", len(X_test))

tree_model = DecisionTreeClassifier(
    max_depth=None,
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)

tree_prediction = tree_model.predict(X_test)

tree_accuracy = accuracy_score(
    y_test,
    tree_prediction
)

print(
    "Decision Tree Accuracy:",
    round(tree_accuracy * 100, 2),
    "%"
)