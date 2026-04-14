import json

import numpy as np
from flask import Flask, Response, request

import ml

app = Flask(__name__)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _json_response(payload, status=200):
    body = json.dumps(_json_safe(payload), default=str)
    return Response(body, status=status, mimetype="application/json")

# Home
@app.route('/')
def home():
    return "Fairness Auditor Backend Running!"

# ==========================================
# 1. RUN PREBUILT PIPELINE (NO INPUT)
# ==========================================
@app.route('/run_pipeline', methods=['GET'])
def run_pipeline():
    result = ml.run_pipeline()

    return _json_response({
        "mode": "demo_pipeline",
        "status": "success",
        "data": result
    })


# ==========================================
# 2. UPLOAD DATASET (BASIC SUPPORT)
# ==========================================
@app.route('/upload_and_run', methods=['POST'])
def upload_and_run():
    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            return _json_response({"status": "error", "message": "No file uploaded."}, 400)

        target = request.form.get("target_column", "").strip()
        if not target:
            return _json_response({"status": "error", "message": "Missing target_column in form data."}, 400)

        raw = file.read()
        df = ml.read_csv_bytes(raw)
        result = ml.run_pipeline_from_dataframe(df, target)

        return _json_response({
            "mode": "custom_csv",
            "status": "success",
            "data": result,
        })

    except ValueError as e:
        return _json_response({"status": "error", "message": str(e)}, 400)
    except Exception as e:
        return _json_response({"status": "error", "message": str(e)}, 500)


if __name__ == '__main__':
    app.run(debug=True)
