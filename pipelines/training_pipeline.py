from src.ml_disease_symptom.steps.data_loader import DataLoader
from src.ml_disease_symptom.steps.data_validator import DataValidator
from src.ml_disease_symptom.steps.data_transformation import DataTransformation
from src.ml_disease_symptom.steps.model_trainer import Model


def training_pipeline(disease, descri, precau):

    dataload = DataLoader()
    train_data, test_data = dataload.data_load(disease, descri, precau)

    datavali = DataValidator()
    train_data_vali, test_data_vali = datavali.data_validator(train_data, test_data)

    datatransform = DataTransformation()
    xtrain, xtest, ytrain, ytest = datatransform.data_transform(
        train_data_vali, test_data_vali
    )

    modeltrainer = Model()
    model = modeltrainer.train_model(xtrain, ytrain)

    modeltrainer.model_evaluate(model, xtest, ytest)
