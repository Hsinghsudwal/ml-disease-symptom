import pandas as pd
import numpy as np
from utils import monitor_data
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score, classification_report
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st


def load_data(path):
    datafile = pd.read_csv(path)
    data = monitor_data(datafile)
    print(data.head())
    print(data.columns)
    #
    return data


def split_data(data):

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    print(train_data.shape, test_data.shape)

    return train_data, test_data


def inputs_model(train_data, test_data):

    xtrain = train_data["Symptom"]
    xtest = test_data["Symptom"]

    ytrain = train_data["Disease"]
    ytest = test_data["Disease"]

    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)

    return xtrain, xtest, ytrain, ytest


def load_model():
    loaded_model = joblib.load(open("./monitor.joblib", "rb"))
    return loaded_model


# def models_drift(model, xtest, ytest):
# y = y.ravel()  # or use y = y.reshape(-1)
# model performance
# model prediction drift
#     import pandas as pd
# from sklearn.metrics import accuracy_score

# def models_drift(model, xtest, ytest):
#     # Convert xtest into a DataFrame, where each row is one test instance

from scipy.sparse import issparse


def models_drift(model, xtest, ytest):

    df = pd.DataFrame(xtest)
    print(df.head())

    # Make predictions on all test samples
    y_pred = model.predict(df)
    print(f"Predictions: {y_pred}")

    # Convert ytest to sparse
    ytest = ytest.to_numpy()
    accuracy = round(accuracy_score(ytest, y_pred), 2)
    print(f"Accuracy: {accuracy}")

    report = classification_report(ytest, y_pred)
    print(f"report: {report}")

    return accuracy, report


def main():

    st.title("Disease Symptom Monitoring")

    path = r"./disease.csv"

    data = load_data(path)

    train_data, test_data = split_data(data)

    xtrain, xtest, ytrain, ytest = inputs_model(train_data, test_data)

    loaded_model = joblib.load(open("./monitor.joblib", "rb"))

    accuracy, report = models_drift(loaded_model, xtest, ytest)

    st.write(f"Model Accuracy: {accuracy}")

    st.write(f"Model Accuracy: {report}")


if __name__ == "__main__":
    main()
