from src.ml_disease_symptom.steps.experiment_mlflow import (
    register_model_to_mlflow,
    staging_model,
    production_stage_archived,
)


def experiment_pipeline():

    register_model_to_mlflow()

    staging_model()

    production_stage_archived()
