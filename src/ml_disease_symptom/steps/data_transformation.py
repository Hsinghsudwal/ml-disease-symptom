import pandas as pd
import os
from src.ml_disease_symptom.utility import *


class DataTransformation:

    def __init__(self) -> None:
        pass

    def data_transform(
        self, train_data_vali: pd.DataFrame, test_data_vali: pd.DataFrame
    ) -> pd.DataFrame:
        try:
            clean_train = data_prep(train_data_vali)
            clean_test = data_prep(test_data_vali)

            param = read_yaml_file("./config/parameters.yml")

            data = param["datas"]["data"]

            transform = param["datas"]["transform"]

            transform_data_path = os.path.join(data, transform)

            os.makedirs(transform_data_path, exist_ok=True)

            clean_train.to_csv(
                os.path.join(transform_data_path, (param["datas"]["train_data_trans"])),
                index=False,
            )
            clean_test.to_csv(
                os.path.join(transform_data_path, (param["datas"]["test_data_trans"])),
                index=False,
            )

            xtrain = clean_train["Symptom"]
            xtest = clean_test["Symptom"]

            ytrain = clean_train["Disease"]
            ytest = clean_test["Disease"]

            # print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)

            print("Data transformation completed")
            return xtrain, xtest, ytrain, ytest

        except Exception as e:
            raise e


