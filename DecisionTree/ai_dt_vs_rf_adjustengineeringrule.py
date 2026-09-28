# =======================================================================================
# AI 人工智慧概論：Data Noise
# 課程目標: 調整engineering_rule的規則，改用「多個條件共同累積風險」的 方式來產生標籤，
# 分析此情況下，Random Forest的預測表現是否能優於Decision Tree
# =======================================================================================

# Step 0：載入套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

# =======================================================================================
# Step 1：產生模擬的機器運轉資料
# =======================================================================================

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

# 加入「無關特徵」，此特徵與故障狀態沒有太直接的關係
data["humidity"] = np.random.normal(60, 10, n)
data["ambient_temperature"] = np.random.normal(25, 5, n)
data["machine_age"] = np.random.uniform(0, 10, n)

# =======================================================================================
# Step 2：加入 Sensor Noise
# =======================================================================================

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

# =======================================================================================
# Step 3：建立 Ground Truth (修改為「多個條件共同累積風險」)
# =======================================================================================

def engineering_rule(row):

    risk_score = 0

    # Temperature
    if row["true_temperature"] > 80:
        risk_score += 1

    if row["true_temperature"] > 90:
        risk_score += 1

    # Vibration
    if row["true_vibration"] > 5:
        risk_score += 1

    if row["true_vibration"] > 7:
        risk_score += 1

    # RPM
    if row["true_rpm"] > 2000:
        risk_score += 1

    if row["true_rpm"] > 2400:
        risk_score += 1

    # Interaction
    if (
        row["true_temperature"] > 82
        and row["true_vibration"] > 5
    ):
        risk_score += 1

    if (
        row["true_rpm"] > 2100
        and row["true_vibration"] > 5
    ):
        risk_score += 1

    # Classification
    if risk_score >= 4:
        return "Danger"

    elif risk_score >= 2:
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

# ============================================================
# Step 4：準備 AI 的輸入 X 和答案 y
# ============================================================

X = data[
    [
        "temperature",
        "vibration",
        "rpm",
        "humidity",
        "ambient_temperature",
        "machine_age"
    ]
]

y = data["true_status"]

# ============================================================
# Step 5：切分 Training Data / Testing Data
# ============================================================


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Training data:", len(X_train))
print("Testing data:", len(X_test))

# ============================================================
# 建立 Decision Tree，並分別計算Training/Testing的準確率
# ============================================================

tree_model = DecisionTreeClassifier(
    max_depth=6,
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)

# Prediction
tree_train_prediction = tree_model.predict(X_train)
tree_test_prediction = tree_model.predict(X_test)

# Accuracy
tree_train_accuracy = accuracy_score(
    y_train,
    tree_train_prediction
)

tree_test_accuracy = accuracy_score(
    y_test,
    tree_test_prediction
)


# ============================================================
# 建立Random Forest，並分別計算Training/Testing的準確率
# ============================================================

forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

forest_model.fit(
    X_train,
    y_train
)

# Prediction
forest_train_prediction = forest_model.predict(X_train)
forest_test_prediction = forest_model.predict(X_test)

# Accuracy
forest_train_accuracy = accuracy_score(
    y_train,
    forest_train_prediction
)

forest_test_accuracy = accuracy_score(
    y_test,
    forest_test_prediction
)


# ============================================================
# Results
# ============================================================

print("====================================")
print("Model Accuracy Comparison")
print("====================================")

print(
    "DT Train Acc.:",
    round(tree_train_accuracy * 100, 2),
    "%"
)

print(
    "DT Test Acc. :",
    round(tree_test_accuracy * 100, 2),
    "%"
)

print(
    "RF Train Acc.:",
    round(forest_train_accuracy * 100, 2),
    "%"
)

print(
    "RF Test Acc. :",
    round(forest_test_accuracy * 100, 2),
    "%"
)