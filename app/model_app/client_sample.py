import requests

# Sample data for prediction
data = {
    "Longitude": -122.23,
    "Latitude": 37.88,
    "Housing_median_age": 41.0,
    "Total_rooms": 880.0,
    "Total_bedrooms": 129.0,
    "Population": 322.0,
    "Households": 126.0,
    "Median_income": 5.3252,
}

url = "http://127.0.0.1:8000/predict/"
response = requests.post(url, json=data)

try:
    if response.status_code == 200:
        result = response.json()
        print("Predicted Median House Value:", result["Median_house_value"])
    else:
        print("Error:", response.status_code)
except requests.exceptions.RequestException as e:
    print("Error connecting to the server:", e)
