import os
import sys

import pandas as pd
import numpy as np
import pytest

PROJECT_ABSOLUTE_PATH = \
    (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ABSOLUTE_PATH)
sys.path.append(os.path.join(PROJECT_ABSOLUTE_PATH, 'src'))

from src.data_processing.preprocess_data import \
    preprocess_train, preprocess_inference, split_dataset, EngineerFeatures
from src.inference import inference

FEAT_COLS = ['Longitude', 'Latitude', 'Housing_median_age', 
             'Total_rooms', 'Total_bedrooms', 'Population', 
             'Households', 'Median_income']
TEST_X_TRAIN = pd.DataFrame(np.random.rand(100, 8), \
                            columns=FEAT_COLS)
TEST_X_VAL = pd.DataFrame(np.random.rand(25, 8), 
                          columns=FEAT_COLS)

TEST_Y_TRAIN_SERIES = pd.Series(np.random.rand(100), 
                         name = 'Median_house_value')
TEST_Y_TRAIN = pd.DataFrame({'Median_house_value': TEST_Y_TRAIN_SERIES})

TEST_Y_VAL_SERIES = pd.Series(np.random.rand(25), 
                       name = 'Median_house_value')
TEST_Y_VAL = pd.DataFrame({'Median_house_value': TEST_Y_VAL_SERIES})

TEST_DF_TRAIN = pd.concat([TEST_X_TRAIN, TEST_Y_TRAIN], axis=1)

TEST_X_TEST = pd.DataFrame(np.random.rand(10, 8), 
                           columns=FEAT_COLS)

def test_preprocess_train():
    X_train_transformed, X_val_transformed = preprocess_train(TEST_X_TRAIN, TEST_X_VAL)
    assert isinstance(X_train_transformed, np.ndarray)
    assert isinstance(X_val_transformed, np.ndarray)
    assert X_train_transformed.shape[0] == TEST_X_TRAIN.shape[0]
    assert X_val_transformed.shape[0] == TEST_X_VAL.shape[0]
    assert X_train_transformed.shape[1] > TEST_X_TRAIN.shape[1]
    assert X_val_transformed.shape[1] > TEST_X_VAL.shape[1]

def test_split_dataset():
    X_train, X_val, y_train, y_val = split_dataset(TEST_DF_TRAIN)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_val, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_val, pd.Series)
    assert X_train.shape[0] + X_val.shape[0] == TEST_X_TRAIN.shape[0]
    assert y_train.shape[0] + y_val.shape[0] == TEST_Y_TRAIN.shape[0]

def test_engineer_features():
    transformer = EngineerFeatures()
    transformed_X = transformer.transform(TEST_X_TRAIN.values)
    assert isinstance(transformed_X, np.ndarray)
    assert transformed_X.shape[0] == TEST_X_TRAIN.shape[0]
    assert transformed_X.shape[1] > TEST_X_TRAIN.shape[1]

def test_inference():
    x_transformed = preprocess_inference(TEST_X_TEST)
    pred = inference(x_transformed)
    assert isinstance(pred, np.ndarray)

if __name__ == "__main__":
    pytest.main()
