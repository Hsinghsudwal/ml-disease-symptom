import streamlit as st

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from utils import data_parse
import joblib


model_file = "./loaded_model.joblib"


datafile = pd.read_csv("./disease.csv")

model = joblib.load(open(model_file, "rb"))


data = data_parse(datafile)


st.title("Disease Symptom Prediction")

st.sidebar.header("DataFrame or Prediction")

selected_tab = st.sidebar.selectbox("Select", ["DataFrame", "Prediction"])

# Update sidebar content based on selected tab
if selected_tab == "DataFrame":
    # st.sidebar.write("DataFrame choose")
    st.write("DataFrame: ")
    st.dataframe(data)


elif selected_tab == "Prediction":
    st.header("Prediction")

    symptom_1 = st.selectbox("Select a symptom", data["Symptom_1"].unique())
    symptom_2 = st.selectbox("Select a symptom", data["Symptom_2"].unique())
    symptom_3 = st.selectbox("Select a symptom", data["Symptom_3"].unique())
    symptom_4 = st.selectbox("Select a symptom", data["Symptom_4"].unique())

    input_data = pd.Series(
        [
            symptom_1,
            symptom_2,
            symptom_3,
            symptom_4,
        ],
        index=[
            "Symptom_1",
            "Symptom_2",
            "Symptom_3",
            "Symptom_4",
        ],
    )

    input_data = " ".join(input_data.fillna(""))

    input_data = input_data.replace("_", " ")

    if st.button("Predict"):

        st.write("Symptom: ", input_data)

        input_df = pd.DataFrame({"Symptom": [input_data]})

        prediction = model.predict(input_df)[0]

        print("prediction: ", prediction)

        st.write(f"Predicted disease: {prediction}")

        decription = data[data["Disease"] == prediction]["Description"].values[0]

        st.write(f"Description: {decription}")

        precaution = data[data["Disease"] == prediction]["Precaution"].values[0]

        st.write(f"Recommendations: {precaution}")
