import os
import sys

import pandas as pd
import joblib

PROJECT_ABSOLUTE_PATH = \
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(f"{PROJECT_ABSOLUTE_PATH}")

MODELS_PATH = os.path.join('res', 'models')
TRAFO_PATH = os.path.join('res', 'transformers')
TRAFO_NAME = 'preprocessor_pipeline.joblib'
DATASET_PATH = \
    os.path.join('data', 'csv', 'CaliforniaHousing', 'california_housing_train.csv')

def load_dataset(csv_path=DATASET_PATH, inference=False):
    """
    Load the dataset from a CSV file.

    Parameters:
        csv_path (str): Path to the CSV file.
        inference (bool): If True, drops the target column "Median_house_value" from the DataFrame.

    Returns:
        pd.DataFrame: Loaded DataFrame containing the dataset.
    """
    dataframe = pd.read_csv(csv_path)
    has_nan = dataframe['Median_house_value'].isna()
    dataframe = dataframe[~has_nan]
    if inference:
        dataframe = dataframe.drop('Median_house_value', axis=1)
    return dataframe

def save_model(model, model_tag='model', model_dir=MODELS_PATH):
    """
    Save a machine learning model to a file.

    Parameters:
        model: The machine learning model to be saved.
        model_tag (str): A custom tag for the model's filename (default: 'model').
        model_dir (str): Directory to save the model (default: MODELS_PATH).

    Returns:
        None
    """
    model_name = f'{model_tag}.joblib'
    model_path = os.path.join(model_dir, model_name)
    joblib.dump(model, model_path)

def load_model(model_path):
    """
    Load a machine learning model from a file.

    Parameters:
        model_path (str): Path to the saved model file.

    Returns:
        The loaded machine learning model.
    """
    model = joblib.load(model_path)
    if model is not None:
        return model
    else:
        raise ValueError(f"Invalid model") 

def save_trafo_pipeline(trafo_pipeline, trafo_dir=TRAFO_PATH):
    """
    Save a transformer pipeline to a file.

    Parameters:
        trafo_pipeline: The transformer pipeline (e.g., Pipeline object containing preprocessing steps).
        trafo_dir (str): Directory to save the transformer pipeline (default: TRAFO_PATH).

    Returns:
        None
    """
    save_path = os.path.join(trafo_dir, TRAFO_NAME)
    joblib.dump(trafo_pipeline, save_path)

def load_trafo_pipeline(trafo_dir=TRAFO_PATH):
    """
    Load a transformer pipeline from a file.

    Parameters:
        trafo_dir (str): Directory where the transformer pipeline is saved (default: TRAFO_PATH).

    Returns:
        The loaded transformer pipeline.
    """
    trafo_path = os.path.join(trafo_dir, TRAFO_NAME)
    trafo = joblib.load(trafo_path)
    if trafo is not None:
        return trafo
    else:
        raise ValueError(f"Invalid transformer") 
