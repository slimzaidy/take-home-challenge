import os
import sys
import argparse

import wandb
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

PROJECT_ABSOLUTE_PATH = \
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ABSOLUTE_PATH)

from utils.utils import load_dataset, save_model, MODELS_PATH
from data_processing.preprocess_data import split_dataset, preprocess_train
from config.sweep_config import sweep_config

DATASET_PATH = \
    os.path.join('data', 'csv', 'CaliforniaHousing', 'california_housing_train.csv')

dataset_df = load_dataset(DATASET_PATH)
X_train, X_val, y_train, y_val = split_dataset(dataset_df)
X_train_transformed, X_val_transformed = preprocess_train(X_train, X_val)
    
def train_and_evaluate(): 
    """
    Train a Linear Regression model and evaluate it using the validation dataset.

    This function trains a Linear Regression model on the preprocessed training data.
    The model is then evaluated on the validation data, and the root mean squared error (RMSE) is calculated.
    The trained model is saved in ./res/models directory
    The model is logged to WandB along with the RMSE value.

    Returns:
        None
    """
    wandb.init(project="boston-housing")

    model = LinearRegression()
    model.fit(X_train_transformed, y_train)
    y_pred = model.predict(X_val_transformed)
    mse = mean_squared_error(y_val, y_pred)
    rmse = round(np.sqrt(mse), 3)
    print(f"rmse is {rmse}")
    model_tag="lin_reg_model"
    wandb.log({'model': 'LinearRegression', 'rmse': rmse})
    save_model(model, model_tag)

    saved_model_path = os.path.join(MODELS_PATH, model_tag + '.joblib')

    model_artifact = wandb.Artifact(model_tag, type="model")
    model_artifact.add_file(saved_model_path)
    wandb.run.log_artifact(model_artifact)

def train_sweep(config=None):
    """
    Train models using different hyperparameter configurations.

    This function is used for hyperparameter search in WandB sweep mode.
    It creates a model using the specified hyperparameters, fits it to the training data, and evaluates it on the validation data.
    The RMSE value is logged along with the model type and hyperparameters.
    The best-performing model is saved and logged as an artifact.
    This sweep is ran with with the GradientBoostingRegressor, but other models can also be used.

    Parameters:
        config (wandb.config): Configuration object from WandB sweep (default: None).

    Returns:
        None
    """
    wandb.init(project="boston-housing", config=sweep_config)
    config = wandb.config

    model_type = config.model
    model = None
    model = GradientBoostingRegressor(learning_rate=config.learning_rate, 
                                        n_estimators=config.n_estimators, 
                                        max_depth=config.max_depth, 
                                        random_state=42)
    if model is not None:
        model.fit(X_train_transformed, y_train)
        y_pred = model.predict(X_val_transformed)
        mse = mean_squared_error(y_val, y_pred)
        rmse = round(np.sqrt(mse), 3)
        wandb.log({'model': model_type, 'rmse': rmse})

        if wandb.run.summary.get('best_rmse') is None or rmse < wandb.run.summary['best_rmse']:
            save_model(model=model, model_tag=config.model)
            wandb.run.summary['best_rmse'] = rmse

            saved_model_path = os.path.join(MODELS_PATH, config.model + '.joblib')
            model_artifact = wandb.Artifact(config.model, type="model")
            model_artifact.add_file(saved_model_path)
            wandb.run.log_artifact(model_artifact)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def log_dataset_artifact_wandb():
    """
    Log the dataset as an artifact to WandB.

    This function logs the dataset DataFrame and the dataset file itself as a WandB artifact.

    Returns:
        None
    """
    boston_table = wandb.Table(dataframe=dataset_df)
    dataset_artifact = wandb.Artifact(f"boston_housing_dataset", type="dataset")
    dataset_artifact.add(boston_table, "boston_table")
    dataset_artifact.add_file(DATASET_PATH)
    wandb.run.log({"boston_table": boston_table})
    wandb.run.log_artifact(dataset_artifact)

if __name__ == '__main__':
    parser = \
        argparse.ArgumentParser(description='Train models on the Boston Housing dataset.')
    parser.add_argument('--sweep', 
                        action='store_true', 
                        help='Enable sweep mode to run hyperparameter search.')
    args = parser.parse_args()

    wandb.login()
    if args.sweep:
        sweep_id = wandb.sweep(sweep_config, project="boston-housing")
        wandb.agent(sweep_id, function=train_sweep)
    else:
        train_and_evaluate()
        log_dataset_artifact_wandb()
    wandb.finish()
