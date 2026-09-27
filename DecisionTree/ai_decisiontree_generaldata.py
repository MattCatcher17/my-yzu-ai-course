# ============================================================
# AI 人工智慧概論：機器運轉資料 + 工程規則 + Decision Tree
# ============================================================

# Step 0：載入套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# Step 1：產生模擬的機器運轉資料
# ============================================================

# 固定亂數種子，讓每次執行結果一致
np.random.seed(42)

# 資料筆數
n = 300

# 模擬感測器資料
data = pd.DataFrame({
    "temperature": np.random.normal(75, 10, n),   # 溫度
    "vibration": np.random.normal(4, 1.5, n),     # 振動
    "rpm": np.random.normal(1800, 300, n)         # 轉速
})

# 避免產生不合理的負值或極端值
data["temperature"] = data["temperature"].clip(40, 110)
data["vibration"] = data["vibration"].clip(0.5, 10)
data["rpm"] = data["rpm"].clip(800, 3000)

print("前 5 筆機器運轉資料：")
display(data.head())

# ============================================================
# Step 2：使用工程規則建立標籤 rule_status
# ============================================================

def engineering_rule(row):

    # 危險狀態
    if (
        row["temperature"] > 90
        or row["vibration"] > 7
        or row["rpm"] > 2500
    ):
        return "Danger"

    # 警告狀態
    elif (
        row["temperature"] > 80
        or row["vibration"] > 5
        or row["rpm"] > 2200
    ):
        return "Warning"

    # 正常狀態
    else:
        return "Normal"


# 將工程規則套用到每一筆資料
data["rule_status"] = data.apply(engineering_rule, axis=1)

print("加入工程狀態後：")
display(data.head(10))

# ============================================================
# Step 3：查看各類別數量
# ============================================================

print(data["rule_status"].value_counts())

# ============================================================
# Step 4：視覺化機器資料
# ============================================================

plt.figure(figsize=(8, 5))

for status in data["rule_status"].unique():

    subset = data[data["rule_status"] == status]

    plt.scatter(
        subset["temperature"],
        subset["vibration"],
        label=status,
        alpha=0.7
    )

plt.xlabel("Temperature")
plt.ylabel("Vibration")
plt.title("Machine Operating Data")
plt.legend()
plt.grid(True)

plt.show()

# ============================================================
# Step 5：準備 AI 的輸入 X 和答案 y
# ============================================================

X = data[
    [
        "temperature",
        "vibration",
        "rpm"
    ]
]

y = data["rule_status"]

print("AI 輸入 X：")
display(X.head())

print("答案 y：")
display(y.head())

# ============================================================
# Step 6：切分 Training Data / Testing Data
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
# Step 7：建立 Decision Tree 模型
# ============================================================

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed.")

# ============================================================
# Step 8：AI 對測試資料進行預測
# ============================================================

prediction = model.predict(X_test)

print("前 10 個 AI 預測：")
print(prediction[:10])

print("真正答案：")
print(y_test.iloc[:10].values)

# ============================================================
# Step 9：計算模型準確率
# ============================================================

accuracy = accuracy_score(
    y_test,
    prediction
)

print("Model accuracy:", accuracy)
print("Accuracy:", round(accuracy * 100, 2), "%")

# ============================================================
# Step 10：Decision Tree 視覺化
# ============================================================

plt.figure(figsize=(18, 9))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree")

plt.show()

model_1 = DecisionTreeClassifier(
    max_depth=1,
    random_state=42
)

# ============================================================
# Step 11：Decision Tree 第一層視覺化
# ============================================================

model_1.fit(X_train, y_train)

plt.figure(figsize=(12, 5))

plot_tree(
    model_1,
    feature_names=X.columns,
    class_names=model_1.classes_,
    filled=True,
    rounded=True,
    fontsize=11
)

plt.title("Step 1: Decision Tree with max_depth = 1")
plt.show()

# ============================================================
# Step 12：Decision Tree 第二層視覺化
# ============================================================

model_2 = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

model_2.fit(X_train, y_train)

plt.figure(figsize=(15, 7))

plot_tree(
    model_2,
    feature_names=X.columns,
    class_names=model_2.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Step 2: Decision Tree with max_depth = 2")
plt.show()

# ============================================================
# Step 13：Decision Tree 第三層視覺化
# ============================================================

model_3 = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model_3.fit(X_train, y_train)

plt.figure(figsize=(18, 9))

plot_tree(
    model_3,
    feature_names=X.columns,
    class_names=model_3.classes_,
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Step 3: Decision Tree with max_depth = 3")
plt.show()