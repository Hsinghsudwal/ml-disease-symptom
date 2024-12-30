from kedro.pipeline import Pipeline, node, pipeline

from .nodes import data_loader, data_prep, data_validation, data_transformation


def create_pipeline(**kwargs) -> Pipeline:
    data_pipeline = pipeline(
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
        ]
    )

    return data_pipeline
