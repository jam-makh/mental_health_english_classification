"""
main_ml.py

Loads pre-cached embeddings from disk (produced by cache_embeddings.py),
then trains and evaluates all ML models against both vectorizers.
Vectorization is not repeated here as it would take too much time.

"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz
from evaluation import cross_validate_model

from evaluation import evaluate_model
from ml.logistic_regression import LogisticRegressionModel
from ml.svm import SVMClassifier


def _load_cache(cache_dir: str) -> dict:
    """
    Load cached embeddings and labels from disk.

    :param cache_dir: Directory written by cache_embeddings.py.
    :type cache_dir: str
    :returns: Cache dictionary with labels and feature matrices.
    :rtype: dict
    """
    p = Path(cache_dir)
    cache = {
        "y_train": joblib.load(p / "y_train.pkl"),
        "y_test":  joblib.load(p / "y_test.pkl"),
        "tfidf": (
            load_npz(str(p / "X_train_tfidf.npz")),
            load_npz(str(p / "X_test_tfidf.npz")),
        ),
    }
    mb_train = p / "X_train_mentalbert.npy"
    mb_test  = p / "X_test_mentalbert.npy"
    if mb_train.exists() and mb_test.exists():
        cache["mentalbert"] = (
            np.load(str(mb_train)),
            np.load(str(mb_test)),
        )
    return cache


def run_ml_pipeline(
    cache_dir: str = "cache",
    labels: list = None,
) -> pd.DataFrame:
    """
    Train and evaluate all ML models using pre-cached embeddings.

    :param cache_dir: Directory produced by cache_embeddings.py.
    :type cache_dir: str
    :param labels: Class label names.
    :type labels: list[str], optional
    :returns: Summary DataFrame with metrics for each model/vectorizer.
    :rtype: pd.DataFrame
    """
    if labels is None:
        labels = ["Anxiety", "Depression", "Suicidal", "Normal"]

    # Load cached label arrays and feature matrices for evaluation.
    cache   = _load_cache(cache_dir)
    y_train = cache["y_train"]
    y_test  = cache["y_test"]

    # Always include TF-IDF; include MentalBERT only when cached embeddings exist.
    vectorizer_sets = {"TF-IDF": cache["tfidf"]}
    if "mentalbert" in cache:
        vectorizer_sets["MentalBERT"] = cache["mentalbert"]

    # Instantiate the model classes used for evaluation.
    models = {
        "Logistic Regression": LogisticRegressionModel(),
        "SVM":                 SVMClassifier(),
    }

    # Accumulate results for all model/vectorizer combinations.
    all_results = []

    # Evaluate each vectorizer's features with every model.
    for vec_name, (X_train, X_test) in vectorizer_sets.items():
        print(f"\n{'=' * 60}")
        print(f"Vectorizer: {vec_name}")
        print("=" * 60)

        for model_name, model in models.items():
            print(f"\n--- {model_name} | {vec_name} ---")

            try:
                # Train the model on the current vectorized training split.
                model.train(X_train, y_train)

                # Measure generalization with cross-validation on training data.
                cv_results = cross_validate_model(model.model, X_train, y_train)

                # Predict both the training and test splits for overfitting checks.
                y_train_pred = model.predict(X_train)
                y_test_pred  = model.predict(X_test)

                # Evaluate metrics and optionally show a confusion matrix.
                result = evaluate_model(
                    y_true=y_test,
                    y_pred=y_test_pred,
                    y_train_true=y_train,
                    y_train_pred=y_train_pred,
                    labels=labels,
                    model_name=f"{model_name} ({vec_name})",
                    show_confusion_matrix=True,
                )

                # Append the result row for this model/vectorizer combination.
                all_results.append({
                    "Model":                model_name,
                    "Vectorizer":           vec_name,
                    # Train
                    "Train Accuracy":       round(result.get("train_accuracy",        float("nan")), 4),
                    "Train Precision (W)":  round(result.get("train_precision_weighted", float("nan")), 4),
                    "Train Recall (W)":     round(result.get("train_recall_weighted",    float("nan")), 4),
                    "Train F1 (W)":         round(result.get("train_f1_weighted",        float("nan")), 4),
                    "Train F1 (Macro)":     round(result.get("train_f1_macro",           float("nan")), 4),
                    # Test
                    "Test Accuracy":        round(result["accuracy"],           4),
                    "Test Precision (W)":   round(result["precision_weighted"], 4),
                    "Test Recall (W)":      round(result["recall_weighted"],    4),
                    "Test F1 (W)":          round(result["f1_weighted"],        4),
                    "Test F1 (Macro)":      round(result["f1_macro"],           4),
                    "CV F1 Macro (mean)": round(cv_results["f1_macro_mean"], 4),
                    "CV F1 Macro (std)":  round(cv_results["f1_macro_std"],  4),
                    "Status":               "OK",
                })

            except Exception as exc:
                # Capture failures without stopping the full pipeline.
                print(f"  [ERROR] {model_name} | {vec_name} failed: {exc}")
                all_results.append({
                    "Model":               model_name,
                    "Vectorizer":          vec_name,
                    "Train Accuracy":      float("nan"),
                    "Train Precision (W)": float("nan"),
                    "Train Recall (W)":    float("nan"),
                    "Train F1 (W)":        float("nan"),
                    "Train F1 (Macro)":    float("nan"),
                    "Test Accuracy":       float("nan"),
                    "Test Precision (W)":  float("nan"),
                    "Test Recall (W)":     float("nan"),
                    "Test F1 (W)":         float("nan"),
                    "Test F1 (Macro)":     float("nan"),
                    "Status":              f"FAILED: {exc}",
                })

    summary_df = pd.DataFrame(all_results)
    print("\n" + "=" * 60)
    print("ML RESULTS SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    return summary_df


if __name__ == "__main__":
    summary = run_ml_pipeline(cache_dir="cache")
    summary.to_csv("ml_results.csv", index=False)
