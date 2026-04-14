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
# 4. CSV LOADING (delimiter detection)
# ==========================================
def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace / BOM from headers so 'Continent' matches '\ufeffContinent' etc."""
    out = df.copy()
    new_cols = []
    for c in out.columns:
        s = str(c).replace("\ufeff", "").strip()
        new_cols.append(s)
    out.columns = new_cols
    return out


def resolve_target_column(df: pd.DataFrame, target_col: str) -> str:
    """Map user-selected target name to actual column (strip, case-insensitive, whitespace-normalized)."""
    if target_col is None or (isinstance(target_col, str) and not str(target_col).strip()):
        raise ValueError("No target column selected.")

    names = list(df.columns)
    if target_col in names:
        return target_col

    t = str(target_col).replace("\ufeff", "").strip()
    if t in names:
        return t

    tl = t.lower()
    for n in names:
        if str(n).strip().lower() == tl:
            return n

    def norm_label(x):
        return " ".join(str(x).split()).lower()

    tn = norm_label(t)
    for n in names:
        if norm_label(n) == tn:
            return n

    preview = ", ".join(map(str, names[:30]))
    more = " …" if len(names) > 30 else ""
    raise ValueError(
        f"Target column '{target_col}' not found after loading the CSV. "
        f"Columns in file: {preview}{more}"
    )


def read_csv_bytes(file_bytes: bytes):
    """
    Load CSV from bytes. Tries delimiter sniffing and common separators so that
    semicolon- or tab-separated exports (common in Excel / EU locales) do not
    collapse into a single column.
    """
    import io

    def _read(**kwargs):
        buf = io.BytesIO(file_bytes)
        return pd.read_csv(buf, **kwargs)

    candidates = []
    # Sniff delimiter (comma vs semicolon vs tab, etc.)
    try:
        df = normalize_dataframe_columns(_read(engine="python", sep=None))
        candidates.append(df)
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass

    for sep in (",", ";", "\t", "|"):
        try:
            df = normalize_dataframe_columns(_read(sep=sep))
            candidates.append(df)
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue

    if candidates:
        return candidates[0]
    return normalize_dataframe_columns(_read())


# ==========================================
# 5. CUSTOM CSV PIPELINE
# ==========================================
def _build_feature_matrix(work: pd.DataFrame, target_col: str):
    """
    Build numeric feature matrix from CSV rows. Handles:
    - Numbers stored as strings (common in CSV exports)
    - Booleans
    - Low-cardinality categorical columns (label-encoded)
    """
    import numpy as np

    if target_col not in work.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")

    X = work.drop(columns=[target_col]).copy()
    if X.shape[1] == 0:
        raise ValueError(
            "No feature columns besides the target. Your CSV may have been read as a single column "
            "(try exporting with comma or semicolon separators), or the file only contains the label column. "
            "Add at least one predictor column, or choose a different target column."
        )

    for col in list(X.columns):
        s = X[col]
        if pd.api.types.is_bool_dtype(s):
            X[col] = s.astype(np.int8)
        elif pd.api.types.is_numeric_dtype(s):
            continue
        else:
            coerced = pd.to_numeric(s, errors="coerce")
            non_null = int(coerced.notna().sum())
            if non_null >= max(1, len(s) // 10):
                X[col] = coerced
            else:
                nuniq = s.nunique(dropna=True)
                if 0 < nuniq <= 25:
                    X[col] = pd.factorize(s.astype(str), sort=True)[0].astype(np.int64)
                else:
                    X = X.drop(columns=[col])

    for col in list(X.columns):
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        nuniq = s.nunique(dropna=True)
        if nuniq == 0:
            X = X.drop(columns=[col])
        elif nuniq <= 25:
            X[col] = pd.factorize(s.astype(str), sort=True)[0].astype(np.int64)
        else:
            X = X.drop(columns=[col])

    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.notna().any(axis=0)]
    X = X.select_dtypes(include=[np.number])
    return X


def run_pipeline_from_dataframe(df: pd.DataFrame, target_col: str):
    """Run the same fairness pipeline on a user CSV (numeric features + target column)."""

    work = normalize_dataframe_columns(df.copy())
    resolved_target = resolve_target_column(work, target_col)
    X = _build_feature_matrix(work, resolved_target)
    if X.shape[1] == 0:
        raise ValueError(
            "No usable feature columns after parsing. "
            "Ensure you have columns besides the target with numbers, or categorical columns with "
            "at most 25 distinct values (they will be encoded automatically)."
        )

    y_raw = work[resolved_target]
    if pd.api.types.is_numeric_dtype(y_raw):
        y_num = pd.to_numeric(y_raw, errors="coerce")
        if y_num.nunique() <= 20:
            y = y_num.round().astype(int)
        else:
            y = (y_num > y_num.median()).astype(int)
    else:
        y = pd.Series(pd.factorize(y_raw.astype(str))[0], index=y_raw.index)

    combined = pd.concat([X, y.rename("__target__")], axis=1)
    combined = combined.dropna()
    X = combined.drop(columns=["__target__"])
    y = combined["__target__"].astype(int)

    if len(X) < 10:
        raise ValueError("Need at least 10 rows after removing missing values.")

    before_dist = pd.Series(y).value_counts().to_dict()
    if len(before_dist) < 2:
        raise ValueError("Target must have at least two classes.")

    before_acc, before_cm, before_report = train_and_evaluate(
        X, y, label="BEFORE SMOTE"
    )

    min_class = pd.Series(y).value_counts().min()
    if min_class < 2:
        raise ValueError("Each class needs at least 2 samples for SMOTE.")
    k_neighbors = min(5, int(min_class) - 1)
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)

    after_dist = pd.Series(y_res).value_counts().to_dict()
    after_acc, after_cm, after_report = train_and_evaluate(
        X_res, y_res, label="AFTER SMOTE"
    )

    return {
        "before_accuracy": float(before_acc),
        "after_accuracy": float(after_acc),
        "before_distribution": before_dist,
        "after_distribution": after_dist,
        "before_confusion_matrix": before_cm.tolist(),
        "after_confusion_matrix": after_cm.tolist(),
        "before_classification_report": before_report,
        "after_classification_report": after_report,
    }


# ==========================================
# 6. MAIN PIPELINE (DEMO)
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