import pandas as pd
import os
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from src.ml_disease_symptom.utility import *
from sklearn.metrics import classification_report
import mlflow
from mlflow.models import infer_signature
import joblib
import json


class Model:

    def __init__(self) -> None:
        pass

    def train_model(self, xtrain: pd.Series, ytrain: pd.Series) -> SklearnPipeline:
        try:
            clf = SklearnPipeline(
                [("tfidf", TfidfVectorizer()), ("gbc", GradientBoostingClassifier())]
            )
            clf.fit(xtrain, ytrain)
            param = read_yaml_file("./config/parameters.yml")

            data = param["datas"]["data"]
            model = param["datas"]["model"]

            model_path = os.path.join(data, model)

            os.makedirs(model_path, exist_ok=True)

            model_name_path = os.path.join(model_path, param["datas"]["model_name"])

            # Save the trained model using joblib
            joblib.dump(clf, model_name_path)

            print("Model trainer completed")
            return clf
        except Exception as e:
            raise e

    def model_evaluate(self, clf, xtest: pd.Series, ytest: pd.Series):
        try:

            param = read_yaml_file("./config/parameters.yml")

            # Set the MLflow server URI and experiment URI
            mlflow.set_tracking_uri(param["server"]["uri"])
            mlflow.set_experiment(param["experiment"]["name"])
            pred = clf.predict(xtest)

            accuracy, precision, recall, f1 = metrics_score(ytest, pred)
            metric_dic = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

            param = read_yaml_file("./config/parameters.yml")

            data = param["datas"]["data"]
            metric = param["datas"]["metics"]

            metric_path = os.path.join(data, metric)
            os.makedirs(metric_path, exist_ok=True)

            metric_name_path = os.path.join(metric_path, param["datas"]["metric_name"])

            with open(metric_name_path, "w") as f_in:
                json.dump(metric_dic, f_in, indent=4)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            report = classification_report(ytest, pred, output_dict=True)
            mlflow.log_dict(report, "classification_report.json")

            # model_name = "ml_disease_symptom"

            mlflow.log_artifact(__file__)

            signature = infer_signature(xtest, clf.predict(xtest))
            mlflow.sklearn.log_model(
                clf, artifact_path=param["model"]["name"], signature=signature
            )

            print(
                f"accuracy-{accuracy}, precision-{precision}, recall-{recall}, f1-{f1}"
            )
            print("Model evaluate completed")

            return clf

        except Exception as e:
            raise e
