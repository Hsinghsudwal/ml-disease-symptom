"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    data_loader,
    data_prep,
    data_validation,
    data_transformation,
    train_model,
    evaluate_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=data_loader,
                inputs=["disease", "description", "precaution"],
                outputs="data_loader_output",
                name="data_loader_node",
            ),
            node(
                func=data_prep,
                inputs="data_loader_output",
                outputs=["train_data_prep", "test_data_prep"],
                name="data_prep_node",
            ),
            node(
                func=data_validation,
                inputs=["train_data_prep", "test_data_prep"],
                outputs=["train_data_validate", "test_data_validate"],
                name="data_validation_node",
            ),
            node(
                func=data_transformation,
                inputs=["train_data_validate", "test_data_validate"],
                outputs=["xtrain", "xtest", "ytrain", "ytest"],
                name="data_transformed_node",
            ),
            node(
                func=train_model,
                inputs=["xtrain", "ytrain"],
                outputs="model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "xtest", "ytest"],
                outputs="metics",
                name="evaluate_model_node",
            ),
        ]
    )
