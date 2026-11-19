"""Train PhishGuard AI.

Example:
python train.py --data dataset/raw/url_features_extracted1.csv --label-column ClassLabel --phishing-value 0.0
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import joblib

from src.data_pipeline import load_dataset, make_splits
from src.feature_extraction import FEATURE_NAMES
from src.model import build_model
from src.evaluation import evaluate_and_save, _best_threshold

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def main():
    parser = argparse.ArgumentParser(description="Train the PhishGuard AI ANN model.")
    parser.add_argument("--data", required=True, help="Path to labelled CSV dataset.")
    parser.add_argument("--url-column", default=None, help="URL column name.")
    parser.add_argument("--label-column", default=None, help="Label column name.")
    parser.add_argument("--phishing-value", default=None, help="Exact raw label representing phishing.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    X, y, dataset_info = load_dataset(
        args.data, args.url_column, args.label_column, args.phishing_value
    )

    if len(X) < 1000:
        raise ValueError(
            f"Dataset has only {len(X)} usable rows. At least 1000 rows are required."
        )

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = make_splits(X, y)

    model = build_model(X_train.shape[1])

    n_legit = int((y_train == 0).sum())
    n_phish = int((y_train == 1).sum())
    class_weights = None
    if n_legit > 0 and n_phish > 0:
        total = n_legit + n_phish
        class_weights = {
            0: total / (2.0 * n_legit),
            1: total / (2.0 * n_phish),
        }

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    val_probs = model.predict(X_val, verbose=0).ravel()
    val_true = y_val.values if hasattr(y_val, "values") else y_val
    tuned_threshold, _ = _best_threshold(val_true, val_probs)

    test_probs = model.predict(X_test, verbose=0).ravel()
    metrics = evaluate_and_save(
        y_test, test_probs, history,
        output_dir=str(RESULTS_DIR),
        threshold=tuned_threshold,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "phishing_model.keras")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    metadata = {
        **dataset_info,
        "feature_names": FEATURE_NAMES,
        "threshold": float(tuned_threshold),
        "positive_class": "phishing",
        "negative_class": "legitimate",
        "test_samples": int(len(y_test)),
        "metrics": metrics,
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\nTraining complete.")
    print("Dataset:", dataset_info)
    print("Tuned threshold:", tuned_threshold)
    print("Test metrics:", json.dumps(metrics, indent=2))
    print(f"Saved model/scaler/metadata in {MODELS_DIR}")
    print(f"Saved evaluation outputs in {RESULTS_DIR}")


if __name__ == "__main__":
    main()