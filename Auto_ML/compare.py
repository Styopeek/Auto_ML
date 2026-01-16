import pandas as pd
import ast
from sklearn.metrics import accuracy_score, recall_score, precision_score

LOG_PATH = "ab_logs.csv"
TRAIN_PATH = "data/train.csv"

# ===============================
# 1. Загрузка логов A/B теста
# ===============================
df = pd.read_csv(
    LOG_PATH,
    header=None,
    names=["timestamp", "model", "prediction", "data"]
)

df["prediction"] = df["prediction"].astype(int)

# ===============================
# 2. Парсинг входных фич
# ===============================
df["input_dict"] = df["data"].apply(ast.literal_eval)
X = pd.json_normalize(df["input_dict"])

df = pd.concat(
    [df.drop(columns=["data", "input_dict"]), X],
    axis=1
)

# ===============================
# 3. Загрузка train.csv
# ===============================
train = pd.read_csv(TRAIN_PATH)

train["Sex"] = train["Sex"].map({"male": 0, "female": 1})

FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]

# ===============================
# 4. Симуляция ground truth
# ===============================
# Используем ближайшее совпадение по ключевым признакам
df = df.merge(
    train[FEATURES + ["Survived"]],
    on=["Pclass", "Sex", "SibSp", "Parch", "Embarked"],
    how="left",
)

df = df.dropna(subset=["Survived"])
df["Survived"] = df["Survived"].astype(int)

print(f"Использовано для оценки: {len(df)} запросов\n")

# ===============================
# 5. Traffic split
# ===============================
print("=== Traffic split ===")
print(df["model"].value_counts(normalize=True))
print()

# ===============================
# 6. Метрики A/B
# ===============================
print("=== Metrics ===")

for model_name in df["model"].unique():
    sub = df[df["model"] == model_name]

    acc = accuracy_score(sub["Survived"], sub["prediction"])
    rec = recall_score(sub["Survived"], sub["prediction"], zero_division=0)
    prec = precision_score(sub["Survived"], sub["prediction"], zero_division=0)

    print(model_name)
    print(f"Accuracy : {acc:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"Precision: {prec:.3f}")
    print("-" * 30)

# ===============================
# 7. Сохранение результата
# ===============================
df.to_csv("ab_logs_with_metrics.csv", index=False)
print("\nФайл ab_logs_with_metrics.csv сохранён")