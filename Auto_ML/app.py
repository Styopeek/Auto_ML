from flask import Flask, request, jsonify
import mlflow
import random
import pandas as pd

app = Flask(__name__)
TRAFFIC_SPLIT = 0.5

prod = mlflow.pyfunc.load_model("models:/titanic_model/Production")
stag = mlflow.pyfunc.load_model("models:/titanic_model/Staging")

@app.route("/predict", methods=["POST"])
def predict():
    data = pd.DataFrame([request.json])

    if random.random() < TRAFFIC_SPLIT:
        model, label = prod, "A"
    else:
        model, label = stag, "B"

    pred = model.predict(data)[0]

    with open("ab_logs.csv", "a") as f:
        f.write(f"{label},{pred},{data.to_dict()}\n")

    return jsonify({"prediction": int(pred), "variant": label})