"""
Evaluation utilities for ML and DL models.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(
    y_true,
    y_pred,
    y_train_true=None,
    y_train_pred=None,
    labels: Optional[List[str]] = None,
    model_name: str = "Model",
    show_confusion_matrix: bool = True,
) -> Dict:
    """
    Evaluate classification performance on test data, and optionally
    on training data to detect overfitting.

    Parameters
    ----------
    y_true : array-like
        Ground truth test labels.
    y_pred : array-like
        Predicted test labels.
    y_train_true : array-like, optional
        Ground truth training labels. If provided alongside
        y_train_pred, training metrics are computed and printed.
    y_train_pred : array-like, optional
        Predicted training labels.
    labels : List[str], optional
        Class label names for display.
    model_name : str, optional
        Model display name used in print headers and plot titles.
    show_confusion_matrix : bool, optional
        Whether to plot the confusion matrix.

    Returns
    -------
    Dict with keys:
        accuracy, precision_weighted, recall_weighted, f1_weighted,
        precision_macro, recall_macro, f1_macro,
        classification_report (dict), report_dataframe (DataFrame),
        and optionally train_* equivalents if training data is provided.
    """
    result = {}

    # ── Test metrics ──────────────────────────────────────────────────
    test_metrics = _compute_metrics(y_true, y_pred, labels)
    result.update(test_metrics)

    print(f"\n{'=' * 60}")
    print(f"Results — {model_name}")
    print("=" * 60)

    if y_train_true is not None and y_train_pred is not None:
        train_metrics = _compute_metrics(y_train_true, y_train_pred, labels)
        result.update({f"train_{k}": v for k, v in train_metrics.items()
                       if k != "classification_report" and k != "report_dataframe"})

        print("\nTrain Metrics:")
        _print_metrics(train_metrics)

    print("\nTest Metrics:")
    _print_metrics(test_metrics)

    print(f"\nClassification Report — {model_name}\n")
    print(test_metrics["report_dataframe"].to_string())

    # ── Confusion matrix ──────────────────────────────────────────────
    if show_confusion_matrix:
        cm = confusion_matrix(y_true, y_pred)
        tick_labels = labels if labels is not None else True

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=tick_labels,
            yticklabels=tick_labels,
        )
        plt.title(f"Confusion Matrix — {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    return result


def _compute_metrics(y_true, y_pred, labels: Optional[List[str]]) -> Dict:
    """
    Compute accuracy, weighted and macro precision/recall/F1,
    plus the full classification report.
    """
    accuracy  = accuracy_score(y_true, y_pred)

    precision_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_w    = recall_score(   y_true, y_pred, average="weighted", zero_division=0)
    f1_w        = f1_score(       y_true, y_pred, average="weighted", zero_division=0)

    precision_m = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_m    = recall_score(   y_true, y_pred, average="macro", zero_division=0)
    f1_m        = f1_score(       y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true, y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()

    return {
        "accuracy":           accuracy,
        "precision_weighted": precision_w,
        "recall_weighted":    recall_w,
        "f1_weighted":        f1_w,
        "precision_macro":    precision_m,
        "recall_macro":       recall_m,
        "f1_macro":           f1_m,
        "classification_report": report,
        "report_dataframe":   report_df,
    }


def _print_metrics(metrics: Dict) -> None:
    """Print accuracy, weighted, and macro metrics in a readable block."""
    print(f"  Accuracy            : {metrics['accuracy']:.4f}")
    print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall    (weighted): {metrics['recall_weighted']:.4f}")
    print(f"  F1        (weighted): {metrics['f1_weighted']:.4f}")
    print(f"  Precision (macro)   : {metrics['precision_macro']:.4f}")
    print(f"  Recall    (macro)   : {metrics['recall_macro']:.4f}")
    print(f"  F1        (macro)   : {metrics['f1_macro']:.4f}")
    