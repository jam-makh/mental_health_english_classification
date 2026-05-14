"""
main_dl.py

Loads pre-cached embeddings from disk (produced by cache_embeddings.py),
then trains and evaluates all DL models against both vectorizers.
Vectorization is never repeated here.

Run order
---------
    1. python cache_embeddings.py
    2. python main_dl.py
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz

from models.evaluation import evaluate_model
from models.dl.lstm import LSTMModel
from models.dl.cnn  import CNNModel
from models.dl.bert import BERTModel


def _load_cache(cache_dir: str) -> dict:
    """
    Load all cached embeddings and labels from disk.

    Parameters
    ----------
    cache_dir : str
        Directory written by cache_embeddings.py.

    Returns
    -------
    dict with keys:
        y_train, y_test,
        tfidf      → (X_train, X_test),
        mentalbert → (X_train, X_test)   only if files exist
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


def run_dl_pipeline(
    cache_dir: str = "cache",
    labels: list = None,
) -> pd.DataFrame:
    """
    Train and evaluate all DL models using pre-cached embeddings.

    Parameters
    ----------
    cache_dir : str
        Directory produced by cache_embeddings.py.
    labels : list[str], optional
        Class label names.

    Returns
    -------
    pd.DataFrame
        One row per (model, vectorizer) with all metrics.
    """
    if labels is None:
        labels = ["Anxiety", "Depression", "Suicidal", "Normal"]

    cache = _load_cache(cache_dir)
    y_train = cache["y_train"]
    y_test  = cache["y_test"]

    vectorizer_sets = {"TF-IDF": cache["tfidf"]}
    if "mentalbert" in cache:
        vectorizer_sets["MentalBERT"] = cache["mentalbert"]

    models = {
        "LSTM": LSTMModel(),
        "CNN":  CNNModel(),
        "BERT": BERTModel(),
    }

    all_results = []

    for vec_name, (X_train, X_test) in vectorizer_sets.items():
        print(f"\n{'=' * 60}")
        print(f"Vectorizer: {vec_name}")
        print("=" * 60)

        for model_name, model in models.items():
            print(f"\n--- {model_name} | {vec_name} ---")

            model.train(X_train, y_train)
            y_pred = model.predict(X_test)

            result = evaluate_model(
                y_true=y_test,
                y_pred=y_pred,
                labels=labels,
                model_name=f"{model_name} ({vec_name})",
                show_confusion_matrix=True,
            )

            macro = result["classification_report"]["macro avg"]
            all_results.append({
                "Model":           model_name,
                "Vectorizer":      vec_name,
                "Accuracy":        round(result["accuracy"], 4),
                "Macro Precision": round(macro["precision"], 4),
                "Macro Recall":    round(macro["recall"], 4),
                "Macro F1":        round(macro["f1-score"], 4),
                "Total Support":   int(macro["support"]),
            })

    summary_df = pd.DataFrame(all_results)
    print("\n" + "=" * 60)
    print("DL RESULTS SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    return summary_df


if __name__ == "__main__":
    summary = run_dl_pipeline(cache_dir="cache")
    summary.to_csv("dl_results.csv", index=False)
