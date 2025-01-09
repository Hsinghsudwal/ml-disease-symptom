import pandas as pd


def data_parse(data):
    data = data.fillna(" ")
    # combine precaution
    data["Precaution"] = data[
        ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
    ].apply(lambda x: ", ".join(x), axis=1)
    # combine symptoms and remove '_' with ' '
    data["Symptom"] = data[
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
    data["Symptom"] = data["Symptom"].str.replace("_", " ")

    return data
