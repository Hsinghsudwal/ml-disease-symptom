# ML Disease Symptom

## Table of Content
- [Problem Statement](#problem-statement)
- [Setup](#setup)
- [Development](#development)
- [Orchestration](#orchestration)
- [Deployment](#deployment)


## Problem Statement

To create a machine learning system that can predict diseases based on the symptoms provided by a user. In healthcare this application can assists in early diagnosis, help make them informed decisions and deliver treatment. Then create streamlit application, use model that is in production to predict disease based on symptoms. Deploy on docker container. The dataset is taken from kaggle [dataset](https://www.kaggle.com/datasets/itachi9604disease-symptom-description-dataset).


## Setup

**local installation: Clone the repository** `git clone https://github.com/Hsinghsudwal/ml-disease-symptom.git`

1. Set up Environment for managing libraries and running python scripts.
   ```bash
   conda create -n venv python==3.11 -y
   ```
2. **Activate environment**
   ```bash
   conda activate venv
   ```

3. **Install Dependencies**:
    Declare any dependencies in `requirements.txt` for `pip` installation.
    To install them, run:
    ```bash
    pip install -r requirements.txt 
   ```

## Development

**Run Jupyter Notebook:** On the terminal, from your main project directory to

   ```
   cd notebook
   jupyter lab
   ```
`notebook.py` to perform data loaded, preprocessing, EDA, model selection: classification problem, model training, model evaluation: model's performance

## Orchestration
**src**:

From jupyter notebook to modular code. To perform pipeline functions: 

which are located `src/project_name/steps` has `data_loader`: to load data from various sources in this case `.csv` file.

`data_prep`: is to combine data sources to one `.csv` file.

`data_validate`: to validate if data feature are within 5% or not (ie. null hypothesis (H0): There is no effect or difference).

`data_transformation`: output train and test data for model selection.
`train_model`: with sklear-pipeline fuction combine feature extraction and ensemble technique to form output model.

`evaluate_model`: with model performance, save the model to mlflow, register the model, stage the model for testing, archived the production model, production the staging model and download the production model that can be used in deployment.

**How to run pipeline:**
   ```bash
   python run.py
   ```

## Deployment
**Creating Docker image and Streamlit application**:

**Steps:** From the main project directory create `directory deployment`. Then `cd deployment`.
1. After running the script `python run.py`, a `disease.csv` file is created in the   `deployment` directory.

2. Create script to run streamlit `app.py`. When completed run via
   ```bash
   Streamlit run app.py
   ```

3. **DOCKER:** From the main project directory `cd deployment`.
* Create docker file for container!
 ```
    FROM python 
    WORKDIR
    COPY
    RUN
    RUN
    EXPOSE
    CMD
```

3. Docker -check image to see if docker is install and/or working on your machine
   ```bash
   docker images
   ```
4. Build Docker
   ```bash
   docker build -t app .
   ```
5. Running Docker 
   ```bash
   docker run it --rm -p 8501:8501 app

   ```
   ```
   8501:8501
   host port:container port
   ```
   or detach running flag -d

   `docker run -it -d --rm -p 8501:8501 app`

## Next Step:
-Monitor (Track Model Performance, Monitor Prediction Drift)
- Best-practices
- Ci/Cd pipline for deployment and monitoring






