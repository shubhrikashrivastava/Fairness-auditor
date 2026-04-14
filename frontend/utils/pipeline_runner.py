"""Run ML pipeline inside the Streamlit process (no Flask required for the dashboard)."""

import sys
from pathlib import Path


def _repo_root() -> Path:
    # frontend/utils -> repo root
    return Path(__file__).resolve().parent.parent.parent


def _import_ml():
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import ml as ml_module  # noqa: PLC0415 - loaded after sys.path fix

    return ml_module


def run_demo_pipeline():
    ml = _import_ml()
    return {"status": "success", "mode": "demo_pipeline", "data": ml.run_pipeline()}


def run_csv_pipeline(file_bytes: bytes, target_column: str):
    ml = _import_ml()
    df = ml.read_csv_bytes(file_bytes)
    data = ml.run_pipeline_from_dataframe(df, target_column)
    return {"status": "success", "mode": "custom_csv", "data": data}


def preview_csv_columns(file_bytes: bytes):
    """Column names exactly as the pipeline will see them (delimiter + header normalization)."""
    ml = _import_ml()
    return list(ml.read_csv_bytes(file_bytes).columns)
