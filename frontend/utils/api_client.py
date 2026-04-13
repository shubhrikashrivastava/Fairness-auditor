import requests

BASE_URL = "http://localhost:5000"

def run_pipeline():
    response = requests.get(f"{BASE_URL}/run_pipeline")
    return response.json()
