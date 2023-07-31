import os

from utils.utils import load_model, load_dataset
from data_processing.preprocess_data import preprocess_inference

DATASET_PATH = \
    os.path.join('data', 'csv', 'CaliforniaHousing', 'california_housing_test.csv')
MODEL_PATH = os.path.join('res', 'models', 'model_lin_reg.joblib')

def inference(X_test, model_path=MODEL_PATH):
    """
    Perform inference on the test data csv file using the pre-trained model.

    Parameters:
        X_test (pd.DataFrame): Test data features DataFrame.
        model_path (str): Path to the pre-trained model file from ./res/models (default: MODEL_PATH).

    Returns:
        None
    """
    model = load_model(model_path)
    y_pred = model.predict(X_test)
    print(f"Prediction is {y_pred} & Prediction shape is {y_pred.shape}")
    return y_pred

if __name__ == '__main__':
    dataset_df = load_dataset(DATASET_PATH, inference=True)
    X_test_transformed = preprocess_inference(dataset_df)
    inference(X_test_transformed)

