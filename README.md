# Kedro: ML Disease Prediction Symptom

## Table of Content
- [Problem Statement](#problem-statement)
- [Setup](#setup)
- [Development](#development)
- [Orchestration](#orchestration)
- [Deployment](#deployment)


## Problem Statement

Edit

## Setup
**Installation: Clone the repository** `git clone repo`

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
**With Kedro and notebooks:**
After installing `requirements.txt`, which containes `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `catalog`, `context`, `pipelines` and `session`.

## Orchestration
**src**:

Framework: Kedro

**How to run your Kedro pipeline**

You can run Kedro project with:
```bash
kedro run --pipeline name
```

## Deployment
Flask or Streamlit







## How to test your Kedro project

Have a look at the files `src/tests/test_run.py` and `src/tests/pipelines/data_science/test_pipeline.py` for instructions on how to write your tests. Run the tests as follows:

```
pytest
```

To configure the coverage threshold, look at the `.coveragerc` file.



