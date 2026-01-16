import mlflow
from mlflow.tracking import MlflowClient

def register(run_id, model_name="titanic_model", stage="Staging"):
    client = MlflowClient()
    uri = f"runs:/{run_id}/model"

    model = mlflow.register_model(uri, model_name)

    client.transition_model_version_stage(
        model_name,
        model.version,
        stage=stage,
        archive_existing=True
    )
