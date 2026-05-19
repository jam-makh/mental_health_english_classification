"""
Evaluation utilities for ML and DL models.
"""

from typing import Dict, List, Optional
import numpy as np
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

from sklearn.model_selection import cross_validate, StratifiedKFold

def cross_validate_model(
    model,
    X,
    y,
    cv_splits: int = 5,
    random_state: int = 42,
) -> Dict:
    """
    Perform stratified k-fold cross-validation for a model.

    :param model: Trained or untrained estimator supporting fit/predict.
    :type model: estimator
    :param X: Input features for cross-validation.
    :type X: array-like
    :param y: Target labels for stratification.
    :type y: array-like
    :param cv_splits: Number of folds for cross-validation.
    :type cv_splits: int
    :param random_state: Random seed for fold shuffling.
    :type random_state: int
    :returns: Dictionary of cross-validation metrics.
    :rtype: Dict
    """
    # Create the stratified cross-validation splitter.
    cv = StratifiedKFold(
        n_splits=cv_splits, shuffle=True, random_state=random_state
    )

    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    # Execute cross-validation across all scoring metrics.
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    results = {
        "accuracy_mean":        np.mean(cv_results["test_accuracy"]),
        "precision_macro_mean": np.mean(cv_results["test_precision_macro"]),
        "recall_macro_mean":    np.mean(cv_results["test_recall_macro"]),
        "f1_macro_mean":        np.mean(cv_results["test_f1_macro"]),
        "f1_macro_std":         np.std(cv_results["test_f1_macro"]),  # useful to report
    }

    print(f"\nCross-Validation ({cv_splits}-fold):")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    return results

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
    Evaluate classification performance on test data and optionally training data.

    :param y_true: Ground truth test labels.
    :type y_true: array-like
    :param y_pred: Predicted test labels.
    :type y_pred: array-like
    :param y_train_true: Ground truth training labels.
    :type y_train_true: array-like, optional
    :param y_train_pred: Predicted training labels.
    :type y_train_pred: array-like, optional
    :param labels: Class label names for display.
    :type labels: List[str], optional
    :param model_name: Model display name used in titles.
    :type model_name: str
    :param show_confusion_matrix: Whether to plot the confusion matrix.
    :type show_confusion_matrix: bool
    :returns: Dictionary containing evaluation metrics and reports.
    :rtype: Dict
    """
    # Store evaluation outputs for return.
    result = {}

    # Test metrics
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

    # Confusion matrix 
    if show_confusion_matrix:
        # Build a confusion matrix to visualize class-level errors.
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
        plt.title(f"Confusion Matrix for {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    return result


def _compute_metrics(y_true, y_pred, labels: Optional[List[str]]) -> Dict:
    """
    Compute accuracy, weighted and macro precision/recall/F1, plus the classification report.

    :param y_true: Ground truth labels.
    :type y_true: array-like
    :param y_pred: Predicted labels.
    :type y_pred: array-like
    :param labels: Optional class label names.
    :type labels: List[str], optional
    :returns: Computed metric values and report objects.
    :rtype: Dict
    """
    # Compute weighted and macro metrics for the output labels.
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
    """
    Print accuracy, weighted, and macro metrics in a readable block.

    :param metrics: Dictionary containing metric values.
    :type metrics: Dict
    :returns: None
    :rtype: None
    """
    print(f"  Accuracy            : {metrics['accuracy']:.4f}")
    print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall    (weighted): {metrics['recall_weighted']:.4f}")
    print(f"  F1        (weighted): {metrics['f1_weighted']:.4f}")
    print(f"  Precision (macro)   : {metrics['precision_macro']:.4f}")
    print(f"  Recall    (macro)   : {metrics['recall_macro']:.4f}")
    print(f"  F1        (macro)   : {metrics['f1_macro']:.4f}")
    