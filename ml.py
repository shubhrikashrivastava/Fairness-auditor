# ==========================================
# FAIRNESS AUDITOR - FINAL IMPROVED VERSION
# ==========================================

import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE


# ==========================================
# 1. LOAD DATASET
# ==========================================
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y


# ==========================================
# 2. CREATE IMBALANCE
# ==========================================
def create_imbalance(X, y):
    X_major = X[y == 0]
    y_major = y[y == 0]

    X_minor = X[y == 1][:50]
    y_minor = y[y == 1][:50]

    X_final = pd.concat([X_major, X_minor])
    y_final = pd.concat([y_major, y_minor])

    return X_final, y_final


# ==========================================
# 3. TRAIN & EVALUATE MODEL
# ==========================================
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_and_evaluate(X, y, label=""):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline: Scaling + Model
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
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return acc, cm, report


# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def run_pipeline():

    # Load data
    X, y = load_data()

    # Create imbalance
    X_imb, y_imb = create_imbalance(X, y)

    # BEFORE SMOTE
    before_dist = y_imb.value_counts().to_dict()
    before_acc, before_cm, before_report = train_and_evaluate(
        X_imb, y_imb, label="BEFORE SMOTE"
    )

    # APPLY SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_imb, y_imb)

    # AFTER SMOTE
    after_dist = pd.Series(y_res).value_counts().to_dict()
    after_acc, after_cm, after_report = train_and_evaluate(
        X_res, y_res, label="AFTER SMOTE"
    )

    # FINAL JSON OUTPUT (Backend Ready)
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


# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    output = run_pipeline()

    print("\n=== FINAL JSON OUTPUT ===")
    print(output)