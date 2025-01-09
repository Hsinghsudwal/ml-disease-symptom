import pandas as pd
import os
from scipy.stats import ks_2samp
from src.ml_disease_symptom.utility import *


class DataValidator:

    def __init__(self) -> None:
        pass

    def data_validator(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> pd.DataFrame:
        try:

            data_results = {}
            for feat in train_data.columns:
                ks_stat, p_value = ks_2samp(train_data[feat], test_data[feat])
                data_results[feat] = p_value
                # print(data_results)

                if p_value < 0.05:
                    return "Error drift data"

            param = read_yaml_file("./config/parameters.yml")

            data = param["datas"]["data"]
            validate = param["datas"]["validate"]

            validate_data_path = os.path.join(data, validate)

            os.makedirs(validate_data_path, exist_ok=True)

            train_data.to_csv(
                os.path.join(validate_data_path, (param["datas"]["train_data_vali"])),
                index=False,
            )
            test_data.to_csv(
                os.path.join(validate_data_path, (param["datas"]["test_data_vali"])),
                index=False,
            )
            print("Data validation completed")
            return train_data, test_data

        except Exception as e:
            raise e
