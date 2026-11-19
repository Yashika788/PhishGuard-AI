"""Load a labelled CSV and create URL-only features."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.feature_extraction import extract_feature_matrix, FEATURE_NAMES

URL_CANDIDATES = ["url", "uri", "website", "link", "domain", "address"]
LABEL_CANDIDATES = ["label", "class", "status", "type", "result", "target", "classlabel"]


def _find_column(columns, candidates, explicit=None):
    if explicit is not None:
        if explicit not in columns:
            raise ValueError(
                f"Column '{explicit}' not found. Available columns: {list(columns)}"
            )
        return explicit
    lowered = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(
        f"Could not identify a column. Available columns: {list(columns)}. "
        "Pass the column name explicitly in the training command."
    )


def _label_to_binary(value, phishing_value=None):
    """Map a raw label to binary: 1 = phishing, 0 = legitimate."""
    text = str(value).strip().lower()

    if phishing_value is not None:
        pv = str(phishing_value).strip().lower()
        try:
            return int(float(text) == float(pv))
        except (ValueError, TypeError):
            return int(text == pv)

    phishing_words = {
        "phishing", "phish", "malicious", "malware", "bad",
        "fraud", "unsafe", "1", "-1",
    }
    legitimate_words = {"legitimate", "legit", "benign", "safe", "good", "0"}

    if text in phishing_words:
        return 1
    if text in legitimate_words:
        return 0

    raise ValueError(
        f"Unrecognised label value '{value}'. Use labels such as phishing/legitimate "
        "or pass --phishing-value with the exact phishing label."
    )


def load_dataset(csv_path, url_column=None, label_column=None, phishing_value=None):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("The CSV file is empty.")

    url_col = _find_column(df.columns, URL_CANDIDATES, url_column)
    label_col = _find_column(df.columns, LABEL_CANDIDATES, label_column)

    df = df[[url_col, label_col]].dropna()
    df[url_col] = df[url_col].astype(str).str.strip()
    df = df[df[url_col].str.len() > 0].drop_duplicates(subset=[url_col])

    df["target"] = df[label_col].apply(
        lambda x: _label_to_binary(x, phishing_value)
    )

    if df["target"].nunique() != 2:
        raise ValueError(
            "Dataset must contain both classes: legitimate (0) and phishing (1)."
        )

    X = extract_feature_matrix(df[url_col].tolist())
    y = df["target"].astype("int32")

    return X[FEATURE_NAMES], y, {
        "url_column": url_col,
        "label_column": label_col,
        "rows": len(df),
    }


def make_splits(X, y, test_size=0.20, val_size=0.20, random_state=42):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=relative_val,
        random_state=random_state, stratify=y_train_full,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler