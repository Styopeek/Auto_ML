import pandas as pd

df = pd.read_csv("data/current.csv")
# Искусственный drift: увеличим Fare
df["Fare"] = df["Fare"] / 10
df.to_csv("data/current.csv", index=False)
print("current.csv обновлён с drift")
