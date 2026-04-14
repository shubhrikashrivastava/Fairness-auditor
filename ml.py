# ==========================================
# FAIRNESS AUDITOR - FINAL VERSION
# ==========================================

import pandas as pd
import io

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

def read_csv_bytes(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

# ==========================================
# 1. LOAD DATA
# ==========================================
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y


# ==========================================
# 2. CREATE IMBALANCE (SAFE)
# ==========================================
def create_imbalance(X, y):

    X_major = X[y == 0]
    y_major = y[y == 0]

    # Keep enough minority samples (important for SMOTE)
    X_minor = X[y == 1][:50]
    y_minor = y[y == 1][:50]

    X_final = pd.concat([X_major, X_minor])
    y_final = pd.concat([y_major, y_minor])

    return X_final, y_final


# ==========================================
# 3. TRAIN & EVALUATE MODEL
# ==========================================
def train_and_evaluate(X, y, label=""):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline: Scaling + Logistic Regression
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

    print("\nClass Distribution BEFORE SMOTE:")
    print(y_imb.value_counts())

    # BEFORE SMOTE
    before_acc, before_cm, before_report = train_and_evaluate(
        X_imb, y_imb, "BEFORE SMOTE"
    )
    before_dist = y_imb.value_counts().to_dict()

    # ==========================================
    # APPLY SMOTE (SAFE VERSION)
    # ==========================================
    smote = SMOTE(random_state=42, k_neighbors=2)

    X_res, y_res = smote.fit_resample(X_imb, y_imb)

    print("\nClass Distribution AFTER SMOTE:")
    print(pd.Series(y_res).value_counts())

    # AFTER SMOTE
    after_acc, after_cm, after_report = train_and_evaluate(
        X_res, y_res, "AFTER SMOTE"
    )
    after_dist = pd.Series(y_res).value_counts().to_dict()

    # ==========================================
    # FINAL JSON OUTPUT
    # ==========================================
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
def run_pipeline_from_dataframe(df, *args, **kwargs):

    import pandas as pd
    from imblearn.over_sampling import SMOTE

    # ==========================================
    # 1. ENCODE CATEGORICAL DATA
    # ==========================================
    df = pd.get_dummies(df, drop_first=True)

    # ==========================================
    # 2. SET TARGET COLUMN
    # ==========================================
    target_col = "loan_paid_back"

    if target_col not in df.columns:
        raise ValueError("Target column 'loan_paid_back' not found")

    # Split
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ==========================================
    # 3. CREATE IMBALANCE
    # ==========================================
    X_imb, y_imb = create_imbalance(X, y)

    # BEFORE SMOTE
    before_acc, before_cm, before_report = train_and_evaluate(
        X_imb, y_imb, "BEFORE SMOTE"
    )
    before_dist = y_imb.value_counts().to_dict()

    # ==========================================
    # 4. APPLY SMOTE
    # ==========================================
    smote = SMOTE(random_state=42, k_neighbors=2)
    X_res, y_res = smote.fit_resample(X_imb, y_imb)

    # AFTER SMOTE
    after_acc, after_cm, after_report = train_and_evaluate(
        X_res, y_res, "AFTER SMOTE"
    )
    after_dist = pd.Series(y_res).value_counts().to_dict()

    # ==========================================
    # 5. RETURN RESULT
    # ==========================================
    return {
        "before_accuracy": float(before_acc),
        "after_accuracy": float(after_acc),
        "before_distribution": before_dist,
        "after_distribution": after_dist,
        "before_confusion_matrix": before_cm.tolist(),
        "after_confusion_matrix": after_cm.tolist(),
        "before_classification_report": before_report,
        "after_classification_report": after_report
    }


# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    output = run_pipeline()

    print("\n=== FINAL OUTPUT ===")
    print(output)