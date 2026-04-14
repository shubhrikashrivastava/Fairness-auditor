import requests

BASE_URL = "http://localhost:5000"


def run_pipeline():
    response = requests.get(f"{BASE_URL}/run_pipeline", timeout=120)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") == "error":
        raise RuntimeError(payload.get("message", "Pipeline failed"))
    return payload


def run_pipeline_upload(file_bytes: bytes, filename: str, target_column: str):
    files = {"file": (filename, file_bytes, "text/csv")}
    data = {"target_column": target_column}
    response = requests.post(
        f"{BASE_URL}/upload_and_run",
        files=files,
        data=data,
        timeout=120,
    )
    if response.status_code >= 400:
        try:
            err = response.json().get("message", response.text)
        except Exception:
            err = response.text
        raise RuntimeError(err or "Upload failed")
    payload = response.json()
    if payload.get("status") == "error":
        raise RuntimeError(payload.get("message", "Upload failed"))
    return payload
