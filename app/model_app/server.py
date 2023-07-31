import os
import sys

from fastapi import FastAPI, HTTPException, Request
import uvicorn
import pandas as pd

PROJECT_ABSOLUTE_PATH = \
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ABSOLUTE_PATH)
sys.path.append(os.path.join(PROJECT_ABSOLUTE_PATH, 'src'))

from src.utils.utils import load_model
from src.data_processing.preprocess_data import preprocess_inference

MODEL_PATH = \
    os.path.join(PROJECT_ABSOLUTE_PATH, 'res', 'models', 'model_lin_reg.joblib')

app = FastAPI(title="Predicting median house price in California")

@app.on_event("startup")
def load_model_on_startup():
    """
    Load the machine learning model when the FastAPI app starts up.
    """
    global model
    model = load_model(MODEL_PATH)

@app.get("/")
def read_root():
    """
    Root endpoint to greet the user.
    """
    return {"Hello": "User"}

@app.post("/predict/")
async def predict_median_house_value(request: Request):
    """
    Predict the median house value based on the provided input data.

    Parameters:
    request (Request): The FastAPI request object.

    Returns:
    dict: A dictionary containing the predicted median house value.
    """
    try:
        data = await request.json()
        input_df = pd.DataFrame.from_dict([data])
        X_test_transformed = preprocess_inference(input_df)
        y_pred = model.predict(X_test_transformed)
        median_house_value = round(y_pred[0], 4)
        return {"Median_house_value": median_house_value}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
