import os
import mlflow
from mlflow.tracking import MlflowClient
from src.ml_disease_symptom.utility import *
import mlflow.pyfunc
import joblib


# Model Registration and Versioning
def register_model_to_mlflow():
    param = read_yaml_file("./config/parameters.yml")
    mlflow.set_tracking_uri(
        param["experiment"]["tracking_uri"]
    )  # "http://127.0.0.1:5000")
    mlflow.set_experiment(param["experiment"]["name"])
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    print(f"Model logged in MLflow with run_id: {mlflow.active_run().info.run_id}")

    # Register the latest model
    model_name = param["model"]["name"]

    model_version = mlflow.register_model(model_uri, model_name)
    print("Model register Completed")

    return model_version, model_name


def staging_model():
    param = read_yaml_file("./config/parameters.yml")
    client = MlflowClient()

    model_name = param["model"]["name"]
    # Register the model in the Model Registry
    registered_model = client.get_registered_model(model_name)

    # Get the latest model version
    latest_version = registered_model.latest_versions[0].version

    # Transition the latest model to the "Staging" stage
    client.transition_model_version_stage(
        name=model_name, version=latest_version, stage="Staging"
    )
    # print("model staging completed")
    print(
        f"Model {model_name} transitioned to Staging stage (version {latest_version})."
    )

    return registered_model, model_name


def production_stage_archived():
    try:
        # Read configuration parameters
        param = read_yaml_file("./config/parameters.yml")

        model_name = param["model"]["name"]
        client = MlflowClient()

        # Get the latest version in "Staging" stage
        print("Getting latest version in Staging stage...")
        staging_versions = client.get_latest_versions(
            name=model_name, stages=["Staging"]
        )

        if not staging_versions:
            raise ValueError("No versions found in the Staging stage.")

        staging_version_number = staging_versions[0].version
        print(f"Staging Version: {staging_version_number}")

        # Get the run ID of the staging model
        run_id = staging_versions[0].run_id
        print("Run ID: ", run_id)

        # Load the model from the staging version
        model_uri = f"runs:/{run_id}/{model_name}"
        print("Model URI:", model_uri)
        loaded_model = mlflow.pyfunc.load_model(model_uri)

        # Define deployment path and save model
        deploy_path = param["deployment"]
        os.makedirs(deploy_path, exist_ok=True)
        deploy_model_path = os.path.join(deploy_path, param["model_deploy"])
        joblib.dump(loaded_model, deploy_model_path)
        print(f"Model saved to {deploy_model_path}.")

        # Define monitor path and save model
        monitor_path = param["monitor"]
        os.makedirs(monitor_path, exist_ok=True)
        monitor_model_path = os.path.join(monitor_path, param["monitor_model"])
        joblib.dump(loaded_model, monitor_model_path)
        print(f"Model saved to {monitor_model_path}.")

        # Transition the current Production model to Archived
        new_production_version = client.get_latest_versions(
            model_name, stages=["Production"]
        )

        if new_production_version:
            production_version_number = new_production_version[0].version
            print(f"Current Production Version: {production_version_number}")

            # Archive the old Production model
            client.transition_model_version_stage(
                name=model_name,
                version=production_version_number,
                stage="archived",
                archive_existing_versions=False,
            )

            # Update the description of the archived model
            client.update_model_version(
                name=model_name,
                version=production_version_number,
                description=f"This model version {production_version_number} is archived.",
            )
            print("Previous Production model archived.")

        # Promote the Staging model to Production
        client.transition_model_version_stage(
            name=model_name,
            version=staging_version_number,
            stage="Production",
            archive_existing_versions=False,
        )

        # Update the description of the newly promoted model
        client.update_model_version(
            name=model_name,
            version=staging_version_number,
            description=f"This model version {staging_version_number} was promoted to Production.",
        )

        print("Staging model promoted to Production.")

    except Exception as e:
        print(f"An error occurred: {e}")


# def production_stage_archived():

#     param = read_yaml_file("./config/parameters.yml")

#     model_name = param["model"]["name"]

#     client = MlflowClient()

#     # Get the latest versions in "Production" and "Staging"
#     print("getting latest version")
#     staging_versions = client.get_latest_versions(name = model_name, stages= ["Staging"])
#     # client.get_latest_versions(name=model_name, stages=)

#     staging_version_number = staging_versions[0].version
#     print(staging_version_number)

#     run_id = staging_versions[0].run_id  # get run ID

#     print("run_id: ", run_id)
#     # Load model from "Staging" version using the run ID
#     model_uri = f"runs:/{run_id}/{model_name}"
#     print(model_uri)
#     loaded_model = mlflow.pyfunc.load_model(model_uri)
#     deploy_path = param["deployment"]
#     os.makedirs(deploy_path, exist_ok=True)
#     deploy_model_path = os.path.join(deploy_path, param["model_deploy"])
#     joblib.dump(loaded_model, deploy_model_path)
#     # joblib.dump(loaded_model, deploy_model_path)
#     print(f"Model saved to {deploy_model_path}.")

#     new_production_version = client.get_latest_versions(model_name, stages=["Production"])

#     if new_production_version:
#         production_version_number = new_production_version[0].version
#         print(production_version_number)

#         # Transition the current Production model to Archived
#         client.transition_model_version_stage(
#             name=model_name,
#             version=production_version_number,
#             stage="archived",
#             archive_existing_versions=False,
#         )


#         client.update_model_version(
#             name=model_name,
#             version=production_version_number,
#             description=f"This model version {production_version_number} is archived.",
#         )
#         print("Model archived")

#     client.transition_model_version_stage(
#         name=model_name,
#         version=staging_version_number,
#         stage="Production",
#         archive_existing_versions=False,
#     )

#     client.update_model_version(
#         name=model_name,
#         version=staging_version_number,
#         description=f"This model version {staging_version_number} was promoted to Production.",
#     )

#     print("Stage Production Complete")
