from flask import Flask, request, jsonify
import pandas as pd
import ml

app = Flask(__name__)

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

    return jsonify({
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
        file = request.files['file']
        df = pd.read_csv(file)

        # Just basic info (since ML doesn't support custom input yet)
        info = {
            "columns": list(df.columns),
            "rows": len(df),
            "preview": df.head().to_dict()
        }

        return jsonify({
            "mode": "custom_dataset",
            "status": "success",
            "message": "Dataset received (ML integration can be extended)",
            "data": info
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })


if __name__ == '__main__':
    app.run(debug=True)
