import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def read_yaml_file(filename):
    with open(filename, "r") as file:
        return yaml.safe_load(file)


def data_prep(data):
    data = data.fillna(" ")
    data["Precaution"] = data[
        ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
    ].apply(lambda x: ", ".join(x), axis=1)
    data["Symptom"] = data[
        ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4", "Symptom_5"]
    ].apply(lambda x: " ".join(x[x.notnull()]), axis=1)
    data["Symptom"] = data["Symptom"].str.replace("_", " ")

    df_data = data[["Disease", "Symptom", "Precaution", "Description"]]

    return df_data


def metrics_score(y_test, y_pred):
    accuracy = round(accuracy_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred, average="weighted"), 2)
    recall = round(recall_score(y_test, y_pred, average="weighted"), 2)
    f1 = round(f1_score(y_test, y_pred, average="weighted"), 2)

    return accuracy, precision, recall, f1
