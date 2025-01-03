"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def data_loader(
    disease: pd.DataFrame, description: pd.DataFrame, precaution: pd.DataFrame
) -> pd.DataFrame:

    disease_desc = pd.merge(disease, description, on="Disease")
    dataset = pd.merge(disease_desc, precaution, on="Disease")

    return dataset


def data_prep(data_loader_output: pd.DataFrame) -> pd.DataFrame:

    # fill na with emty
    data_loader_output = data_loader_output.fillna(" ")

    # combine precaution
    data_loader_output["Precaution"] = data_loader_output[
        ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
    ].apply(lambda x: ", ".join(x), axis=1)

    # combine symptoms and remove '_' with ' '
    data_loader_output["Symptom"] = data_loader_output[
        [
            "Symptom_1",
            "Symptom_2",
            "Symptom_3",
            "Symptom_4",
            "Symptom_5",
            "Symptom_6",
            "Symptom_7",
            "Symptom_8",
            "Symptom_9",
            "Symptom_10",
            "Symptom_11",
            "Symptom_12",
            "Symptom_13",
            "Symptom_14",
            "Symptom_15",
            "Symptom_16",
            "Symptom_17",
        ]
    ].apply(lambda x: " ".join(x[x.notnull()]), axis=1)

    data_loader_output["Symptom"] = data_loader_output["Symptom"].str.replace("_", " ")

    # select only working data
    df_data = data_loader_output[["Disease", "Symptom", "Precaution", "Description"]]

    # split data
    train_data_prep, test_data_prep = train_test_split(
        df_data, test_size=0.2, random_state=42
    )
    print(train_data_prep.shape, test_data_prep.shape)

    return train_data_prep, test_data_prep


def data_validation(
    train_data_prep: pd.DataFrame, test_data_prep: pd.DataFrame
) -> pd.DataFrame:

    data_results = {}

    for feat in train_data_prep.columns:

        ks_stat, p_value = ks_2samp(train_data_prep[feat], test_data_prep[feat])

        data_results[feat] = p_value
        # print(f'{feat}, : {p_value}')
        if p_value < 0.05:
            # print(f"Data drift detected in feature: {feat} (p-value: {p_value})")
            return "Error drift data"

        else:
            return train_data_prep, test_data_prep


def data_transformation(
    train_data_prep: pd.DataFrame, test_data_prep: pd.DataFrame
) -> pd.DataFrame:

    xtrain = train_data_prep["Symptom"]
    xtest = test_data_prep["Symptom"]

    ytrain = train_data_prep["Disease"]
    ytest = test_data_prep["Disease"]
    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)

    return xtrain, xtest, ytrain, ytest


def metrics_score(y_test, y_pred):
    accuracy = round(accuracy_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred, average="weighted"), 2)
    recall = round(recall_score(y_test, y_pred, average="weighted"), 2)
    f1 = round(f1_score(y_test, y_pred, average="weighted"), 2)

    return accuracy, precision, recall, f1


def train_model(xtrain: pd.Series, ytrain: pd.Series) -> tuple:
    """
    Train the model and log it to MLflow, transition to Staging.
    """
    print(xtrain.shape, ytrain.shape)

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("gbc", GradientBoostingClassifier()),
        ]
    )

    model.fit(xtrain, ytrain)

    print(model)

    return model


def evaluate_model(model: Pipeline, xtest: pd.Series, ytest: pd.Series):
    """
    Evaluate the model on test data and log the evaluation metrics.
    """
    pred = model.predict(xtest)
    accuracy, precision, recall, f1 = metrics_score(ytest, pred)

    report = classification_report(ytest, pred, output_dict=True)

    print(f"accuracy-{accuracy}, precision-{precision}, recall-{recall}, f1-{f1}")

    # return accuracy
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
