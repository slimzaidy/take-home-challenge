import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

PROJECT_ABSOLUTE_PATH = \
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(f"{PROJECT_ABSOLUTE_PATH}")
from src.utils.utils import save_trafo_pipeline, load_trafo_pipeline

FEATURE_COLUMNS = ['Longitude', 'Latitude', 'Housing_median_age', 
                'Total_rooms','Total_bedrooms', 'Population', 
                'Households', 'Median_income']

TARGET_COLUMN = 'Median_house_value'

class EngineerFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer to engineer new features from existing ones.

    This transformer calculates additional features based on the existing features.
    """
    def fit(self, X, y= None):
        return self
    
    def transform(self, X, y=None):
        idx_total_rooms = 3
        idx_total_bedrooms = 4
        idx_housholds = 6
        idx_median_income = 7
        idx_housing_med_age = 2

        Total_rooms_per_Households = X[:, idx_total_rooms] / X[:, idx_housholds]
        Median_income_per_household = X[:, idx_median_income] /X [:, idx_housholds]
        Housing_median_age_per_Total_rooms = X[:, idx_housing_med_age] / X[:, idx_total_rooms]
        Median_income_per_Total_bedrooms = X[:, idx_median_income] / X[:, idx_total_bedrooms]
        Median_income_per_Total_rooms = X[:, idx_median_income] / X[:, idx_total_rooms]

        return np.c_[X, Total_rooms_per_Households, Median_income_per_household,
                     Housing_median_age_per_Total_rooms, Median_income_per_Total_bedrooms,
                     Median_income_per_Total_rooms]
    
def split_dataset(dataframe):
    """
    Split the input DataFrame into train and validation sets.

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame containing both features and target column.

    Returns:
        tuple: Four DataFrames representing X_train, X_val, y_train, and y_val.
    """
    X_train, X_val, y_train, y_val = train_test_split(dataframe[FEATURE_COLUMNS],
                                                      dataframe[TARGET_COLUMN],
                                                      test_size=0.1,
                                                      random_state=42)

    return X_train, X_val, y_train, y_val


def preprocess_train(X_train, X_val):
    """
    Preprocess the training and validation sets.

    This function applies imputation, custom feature engineering, and scaling to the data.

    Parameters:
        X_train (pd.DataFrame): Training set features DataFrame.
        X_val (pd.DataFrame): Validation set features DataFrame.

    Returns:
        tuple: Transformed DataFrames for X_train and X_val.
    """
    trafo_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),
                               ('custom feature engineering', EngineerFeatures()),
                               ('scaler', RobustScaler())])
    X_train_transformed = trafo_pipeline.fit_transform(X_train)
    X_val_transformed = trafo_pipeline.transform(X_val)
    save_trafo_pipeline(trafo_pipeline=trafo_pipeline)
    return X_train_transformed, X_val_transformed

def preprocess_inference(X_test):
    """
    Preprocess the inference data (test set).

    This function loads the prefitted transformer pipeline and applies it to the test set.

    Parameters:
        X_test (pd.DataFrame): Test set features DataFrame.

    Returns:
        pd.DataFrame: Transformed test set features DataFrame.
    """
    trafo_pipeline = load_trafo_pipeline()
    X_test_transformed = trafo_pipeline.transform(X_test)
    return X_test_transformed