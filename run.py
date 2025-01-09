import pandas as pd

from pipelines.training_pipeline import training_pipeline
from pipelines.experiment_pipeline import experiment_pipeline


def main():

    disease = pd.read_csv(r"./data_source/disease.csv")  # ,sep=',')
    descri = pd.read_csv(r"./data_source/symptom_description.csv")  # ,sep=',')
    precau = pd.read_csv(r"./data_source/symptom_precaution.csv")  # ,sep=',')

    training_pipeline(disease, descri, precau)

    experiment_pipeline()


if __name__ == "__main__":
    main()
