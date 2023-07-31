import os

import pandas as pd
from sklearn.model_selection import train_test_split

DIR = os.path.join("data", "csv", "CaliforniaHousing")

def split_and_save_data(input_file, 
                        train_file, 
                        test_file, 
                        test_size=0.05, 
                        random_state=42):
    """
    Split the input CSV data file into train and test sets and save them as separate CSV files.

    Parameters:
        input_file (str): Filename of the input CSV data file.
        train_file (str): Filename of the output CSV file for the training set.
        test_file (str): Filename of the output CSV file for the test set.
        test_size (float): The proportion of the dataset to include in the test split (default: 0.05).
        random_state (int): Random seed for reproducibility (default: 42).

    Returns:
        None
    """
    input_path = os.path.join(DIR, input_file)
    train_file = os.path.join(DIR, train_file)
    test_file = os.path.join(DIR, test_file)

    data = pd.read_csv(input_path)

    train_data, test_data = \
        train_test_split(data, 
                         test_size=test_size, 
                         random_state=random_state)

    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

if __name__ == "__main__":
    input_file = "california_housing.csv" 
    train_file = "california_housing_train.csv" 
    test_file = "california_housing_test.csv"  

    split_and_save_data(input_file, train_file, test_file)
