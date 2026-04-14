# ==========================================
# FAIRNESS AUDITOR - FINAL COMPLETE VERSION
# ==========================================

import pandas as pd
import io

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE


# ==========================================
# READ CSV FROM FRONTEND
# ==========================================
def read_csv_bytes(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))


# ==========================================
# CREATE ARTIFICIAL IMBALANCE
# ==========================================
def create_imbalance(X, y):
    X_major = X[y == 0]
    y_major = y[y == 0]

    X_minor = X[y == 1][:50]
    y_minor = y[y == 1][:50]

    return pd.concat([X_major, X_minor]), pd.concat([y_major, y_minor])


# ==========================================
# TRAIN & EVALUATE MODEL
# ==========================================
def train_and_evaluate(X, y, label=""):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n--- {label} ---")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)

    return acc, cm, report


# ==========================================
# MAIN PIPELINE (WORKS FOR ANY DATASET)
# ==========================================
def run_pipeline_from_dataframe(df, *args, **kwargs):

    try:
        # 1. Clean data
        df = df.dropna()

        # 2. Detect target column
        possible_targets = ["target", "label", "class", "output"]

        target_col = None
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break

        if target_col is None:
            target_col = df.columns[-1]

        print(f"Using '{target_col}' as target")

        # 3. Convert target if needed
        if df[target_col].dtype == "object":
            df[target_col] = df[target_col].astype("category").cat.codes

        # 4. Encode categorical features
        df = pd.get_dummies(df, drop_first=True)

        # 5. Split
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 6. Create imbalance
        X_imb, y_imb = create_imbalance(X, y)

        before_dist = y_imb.value_counts().to_dict()

        # BEFORE SMOTE
        before_acc, before_cm, before_report = train_and_evaluate(
            X_imb, y_imb, "BEFORE SMOTE"
        )

        # 7. Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=2)
        X_res, y_res = smote.fit_resample(X_imb, y_imb)

        after_dist = pd.Series(y_res).value_counts().to_dict()

        # AFTER SMOTE
        after_acc, after_cm, after_report = train_and_evaluate(
            X_res, y_res, "AFTER SMOTE"
        )

        # 8. Final result (IMPORTANT: all keys present)
        result = {
            "before_accuracy": float(before_acc),
            "after_accuracy": float(after_acc),
            "before_distribution": before_dist,
            "after_distribution": after_dist,
            "before_confusion_matrix": before_cm.tolist(),
            "after_confusion_matrix": after_cm.tolist(),
            "before_classification_report": before_report,
            "after_classification_report": after_report
        }

        return result

    except Exception as e:
        print("ERROR:", str(e))
        return {
            "before_accuracy": 0,
            "after_accuracy": 0,
            "before_distribution": {},
            "after_distribution": {},
            "before_confusion_matrix": [],
            "after_confusion_matrix": [],
            "before_classification_report": {},
            "after_classification_report": {},
            "error": str(e)
        }


# ==========================================
# MAIN ENTRY FUNCTION (FRONTEND CALLS THIS)
# ==========================================
def run_pipeline(file_bytes=None, *args, **kwargs):

    try:
        # Case 1: CSV uploaded
        if file_bytes is not None:
            df = read_csv_bytes(file_bytes)
            return run_pipeline_from_dataframe(df)

        # Case 2: No CSV → use default dataset
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target

        return run_pipeline_from_dataframe(df)

    except Exception as e:
        print("ERROR:", str(e))
        return {
            "before_accuracy": 0,
            "after_accuracy": 0,
            "before_distribution": {},
            "after_distribution": {},
            "before_confusion_matrix": [],
            "after_confusion_matrix": [],
            "before_classification_report": {},
            "after_classification_report": {},
            "error": str(e)
        }