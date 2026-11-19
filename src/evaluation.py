"""Evaluation metrics and reproducible plots."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


def _best_threshold(y_true, probabilities):
    """Find threshold that maximizes F1 on the positive (phishing) class."""
    best_t, best_f1 = 0.5, -1.0
    for t in [i / 100 for i in range(20, 81)]:
        preds = (probabilities >= t).astype(int)
        f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def evaluate_and_save(
    y_true,
    probabilities,
    history,
    output_dir="results",
    threshold=None,
):
    """
    y_true: 0 = Legitimate, 1 = Phishing
    probabilities: model output (phishing probability)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    y_true = y_true.values if hasattr(y_true, "values") else y_true
    probabilities = probabilities.ravel()

    tuned_threshold, tuned_f1 = _best_threshold(y_true, probabilities)
    decision_threshold = threshold if threshold is not None else tuned_threshold
    predictions = (probabilities >= decision_threshold).astype(int)

    phishing_precision = precision_score(
        y_true, predictions, pos_label=1, zero_division=0
    )
    phishing_recall = recall_score(
        y_true, predictions, pos_label=1, zero_division=0
    )
    phishing_f1 = f1_score(
        y_true, predictions, pos_label=1, zero_division=0
    )

    legit_precision = precision_score(
        y_true, predictions, pos_label=0, zero_division=0
    )
    legit_recall = recall_score(
        y_true, predictions, pos_label=0, zero_division=0
    )
    legit_f1 = f1_score(
        y_true, predictions, pos_label=0, zero_division=0
    )

    metrics = {
        "threshold": float(decision_threshold),
        "tuned_threshold": float(tuned_threshold),
        "tuned_f1": float(tuned_f1),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "phishing": {
            "precision": float(phishing_precision),
            "recall": float(phishing_recall),
            "f1_score": float(phishing_f1),
            "support": int((y_true == 1).sum()),
        },
        "legitimate": {
            "precision": float(legit_precision),
            "recall": float(legit_recall),
            "f1_score": float(legit_f1),
            "support": int((y_true == 0).sum()),
        },
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        metrics["roc_auc"] = None

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    (out / "classification_report.txt").write_text(
        classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Phishing"],
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
    )
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("PhishGuard AI – Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close()

    if history is not None:
        hist = history.history
        for metric, title, filename in [
            ("accuracy", "Model Accuracy", "training_accuracy.png"),
            ("loss", "Model Loss", "training_loss.png"),
        ]:
            if metric in hist:
                plt.figure(figsize=(7, 4))
                plt.plot(hist[metric], label=f"Training {metric}")
                val_key = f"val_{metric}"
                if val_key in hist:
                    plt.plot(hist[val_key], label=f"Validation {metric}")
                plt.title(title)
                plt.xlabel("Epoch")
                plt.ylabel(metric.capitalize())
                plt.legend()
                plt.tight_layout()
                plt.savefig(out / filename, dpi=160)
                plt.close()

    return metrics