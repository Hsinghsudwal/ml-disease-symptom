import pandas as pd
import pandas as pd
import os
from sklearn.model_selection import train_test_split


from src.ml_disease_symptom.utility import *


class DataLoader:

    def __init__(self) -> None:
        pass

    def data_load(
        self, disease: pd.DataFrame, description: pd.DataFrame, precaution: pd.DataFrame
    ) -> pd.DataFrame:
        try:

            disease_desc = pd.merge(disease, description, on="Disease")
            dataset = pd.merge(disease_desc, precaution, on="Disease")

            train_data_set, test_data_set = train_test_split(
                dataset, test_size=0.2, random_state=42
            )
            # print(train_data_set.head())
            # '..config\parameters.yml'
            param = read_yaml_file("./config/parameters.yml")

            data = param["datas"]["data"]
            raw = param["datas"]["raw"]

            raw_data_path = os.path.join(data, raw)

            # Creating directories: exist
            os.makedirs(raw_data_path, exist_ok=True)

            train_data_set.to_csv(
                os.path.join(raw_data_path, (param["datas"]["train_data"])), index=False
            )
            test_data_set.to_csv(
                os.path.join(raw_data_path, (param["datas"]["test_data"])), index=False
            )

            # Define deployment path and save data
            deploy_path = param["deployment"]
            os.makedirs(deploy_path, exist_ok=True)
            dataset.to_csv(os.path.join(deploy_path, (param["disease"])), index=False)

            # Define deployment path and save data
            monitor_path = param["monitor"]
            os.makedirs(monitor_path, exist_ok=True)
            dataset.to_csv(os.path.join(monitor_path, (param["disease_csv"])), index=False)

            print("Data loaded completed")
            return train_data_set, test_data_set
        except Exception as e:
            raise e
