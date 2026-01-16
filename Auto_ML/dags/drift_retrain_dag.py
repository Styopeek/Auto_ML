from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime
import pandas as pd

from drift.psi import dataset_psi
from training.pycaret_automl import train_automl
from mlflow_utils.register import register

TRAIN = "data/train.csv"
CURRENT = "data/current.csv"

def check_drift():
    drift, scores = dataset_psi(
        pd.read_csv(TRAIN),
        pd.read_csv(CURRENT)
    )
    print(scores)
    return "retrain" if drift else "no_drift"

def retrain():
    run_id = train_automl(TRAIN, CURRENT)
    register(run_id)

def no_drift():
    print("Drift not detected")

with DAG(
    "ml_drift_retraining",
    start_date=datetime(2024,1,1),
    schedule_interval=None,
    catchup=False
) as dag:

    check = BranchPythonOperator(
        task_id="check_drift",
        python_callable=check_drift
    )

    retrain_task = PythonOperator(
        task_id="retrain",
        python_callable=retrain
    )

    skip = PythonOperator(
        task_id="no_drift",
        python_callable=no_drift
    )

    check >> [retrain_task, skip]
