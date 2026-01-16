import mlflow
import pandas as pd
from pycaret.classification import *

def train_automl(train_path, current_path=None):
    df = pd.read_csv(train_path)

    if current_path:
        df = pd.concat([df, pd.read_csv(current_path)])

    setup(
        df,
        target="Survived",
        session_id=42,
        log_experiment=True,
        experiment_name="titanic_automl"
    )

    best = compare_models()
    final = finalize_model(best)

    with mlflow.start_run(run_name="pycaret_best_model") as run:
        mlflow.pycaret.log_model(final, "model")
        return run.info.run_id
