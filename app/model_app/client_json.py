import requests
import json

json_file_path = "app/model_app/exp.json"

with open(json_file_path, "r") as file:
    json_data = file.read()

data = json.loads(json_data)
response = requests.post("http://127.0.0.1:8000/predict/", json=data)

if response.status_code == 200:
    result = response.json()
    print(result)
else:
    print("Prediction request failed.")
