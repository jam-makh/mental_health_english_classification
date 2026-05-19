"""
main_ml.py

Train and evaluate all ML models on all vectorizers.

Pairing:
    TF-IDF      → LogisticRegression, SVM, TorchMultinomialNaiveBayes
    MentalBERT  → LogisticRegression, SVM, TorchGaussianNaiveBayes

Saved artefacts
---------------
    results/models/<model>_<vectorizer>.pt   ← Torch NB models
    results/models/<model>_<vectorizer>.pkl  ← sklearn-style models
    results/final_model_results.csv          ← joint metrics table
    results/<name>_confusion_matrix.png      ← one per (model, vec, split)
"""

from pathlib import Path
from typing import Dict, List, Optional
import pickle

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz

from ml.logistic_regression import LogisticRegressionModel
from ml.svm import SVMClassifier
from ml.TorchMultinomialNaiveBayes import TorchMultinomialNaiveBayes
from ml.TorchGaussianNaiveBayes import TorchGaussianNaiveBayes

from evaluation import (
    evaluate_model,
    save_confusion_matrix,
)

# Device — single source of truth, proper torch.device object
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# Cache loading
def load_cache(cache_dir: str) -> dict:
    """
    Load pre-computed embeddings and labels from disk.

    :param cache_dir: Directory produced by cache_embeddings.py.
    :type cache_dir: str
    :returns: Dict with keys y_train, y_test, tfidf, mentalbert (optional).
    :rtype: dict
    """
    base_dir = Path(__file__).resolve().parent
    p = (
        (base_dir / cache_dir).resolve()
        if not Path(cache_dir).is_absolute()
        else Path(cache_dir)
    )

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


# Model factory — vectorizer-aware pairings
def make_models(vectorizer_name: str, allow_mentalbert_lr: bool = True) -> Dict[str, object]:
    """
    Return the models that should run for a given vectorizer.

    TF-IDF    : LogisticRegression, SVM, TorchMultinomialNaiveBayes
    MentalBERT: LogisticRegression (optional), SVM, TorchGaussianNaiveBayes

    :param vectorizer_name: One of ``"TF-IDF"`` or ``"MentalBERT"``.
    :type vectorizer_name: str
    :param allow_mentalbert_lr: Whether to include Logistic Regression on MentalBERT.
    :type allow_mentalbert_lr: bool
    :returns: Ordered dict of {model_name: model_instance}.
    :rtype: Dict[str, object]
    """
    models: Dict[str, object] = {
        "SVM": SVMClassifier(),
    }

    if vectorizer_name == "TF-IDF":
        models["Logistic Regression"] = LogisticRegressionModel()
        # Multinomial NB: sparse non-negative TF-IDF input, GPU for matmul.
        models["Multinomial NB"] = TorchMultinomialNaiveBayes(
            alpha=1.0,
            dtype=torch.float32,
            device=str(DEVICE),   # TorchMultinomialNaiveBayes stores device as str
        )

    elif vectorizer_name == "MentalBERT":
        if allow_mentalbert_lr:
            models["Logistic Regression"] = LogisticRegressionModel()
        # Gaussian NB: dense continuous embeddings, CPU-only by design.
        models["Gaussian NB"] = TorchGaussianNaiveBayes(
            var_smoothing=1e-9,
            dtype=torch.float32,
        )

    return models


# Label encoding
def _encode_labels(y, label_list: List[str]) -> np.ndarray:
    """
    Map string labels to integer indices matching label_list order.

    :param y: Iterable of string labels.
    :param label_list: Ordered list of all class names.
    :returns: Integer array of shape (n,).
    :rtype: np.ndarray
    """
    mapping = {lbl: i for i, lbl in enumerate(label_list)}
    return np.array([mapping[lbl] for lbl in y], dtype=np.int64)

# Training dispatch
def train_model(
    model: object,
    X_train,
    y_train: np.ndarray,
    n_classes: int,
) -> None:
    """
    Dispatch training to the correct model API.

    sklearn-style models (LR, SVM) expose ``.train(X, y)``.
    Torch NB models expose ``.fit(X_tensor, y_tensor, n_classes)``.

    :param model: Model instance.
    :param X_train: Feature matrix — scipy sparse for TF-IDF, ndarray for MentalBERT.
    :param y_train: Integer label array (already encoded).
    :type y_train: np.ndarray
    :param n_classes: Total number of classes.
    :type n_classes: int
    """
    if isinstance(model, TorchMultinomialNaiveBayes):
        # scipy sparse CSR → torch sparse COO, moved to DEVICE
        coo = X_train.tocoo()
        indices = torch.tensor(
            np.vstack((coo.row, coo.col)), dtype=torch.long
        )
        values   = torch.tensor(coo.data, dtype=torch.float32)
        X_tensor = torch.sparse_coo_tensor(
            indices, values, coo.shape,
            dtype=torch.float32,
            device=DEVICE,
        )
        y_tensor = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
        model.fit(X_tensor, y_tensor, n_classes=n_classes)

    elif isinstance(model, TorchGaussianNaiveBayes):
        # Dense ndarray → CPU tensors (Gaussian NB is CPU-only by design)
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)
        model.fit(X_tensor, y_tensor, n_classes=n_classes)

    else:
        # LogisticRegressionModel / SVMClassifier
        model.train(X_train, y_train)


# Predict wrapper — makes Torch NB models look sklearn-compatible
class _TorchModelWrapper:
    """
    Thin adapter so fitted Torch NB models can be passed to
    ``evaluate_model()`` without any changes to that function.

    Handles scipy sparse → torch sparse (Multinomial) and
    ndarray → torch dense (Gaussian) conversion automatically.
    Converts integer predictions back to string labels for consistency.
    """

    def __init__(
        self,
        torch_model,
        is_sparse: bool = False,
        label_map: Optional[Dict[int, str]] = None,
    ) -> None:
        self._model     = torch_model
        self._is_sparse = is_sparse
        self._label_map = label_map  # Maps int index → string label

    def predict(self, X) -> np.ndarray:
        if self._is_sparse:
            coo     = X.tocoo()
            indices = torch.tensor(
                np.vstack((coo.row, coo.col)), dtype=torch.long
            )
            values  = torch.tensor(coo.data, dtype=torch.float32)
            X_t     = torch.sparse_coo_tensor(
                indices, values, coo.shape,
                dtype=torch.float32,
                device=DEVICE,
            )
        else:
            # Gaussian NB is CPU-only no device needed
            X_t = torch.tensor(X, dtype=torch.float32)

        preds = self._model.predict(X_t).cpu().numpy()

        # Convert integer predictions back to string labels
        if self._label_map is not None:
            preds = np.array([self._label_map[p] for p in preds])

        return preds


# models saved under results/models/

RESULTS_DIR = Path("results")
MODELS_DIR  = RESULTS_DIR / "models"


def save_model(
    model: object,
    model_name: str,
    vectorizer_name: str,
) -> Path:
    """
    Persist a trained model to ``results/models/``.

    Torch NB models use their own ``.save()`` → ``.pt`` file.
    sklearn-style models are pickled → ``.pkl`` file.

    :param model: Fitted model instance.
    :param model_name: Human-readable model name (used in filename).
    :type model_name: str
    :param vectorizer_name: Vectorizer name (used in filename).
    :type vectorizer_name: str
    :returns: Path to the saved file.
    :rtype: Path
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    stem = f"{model_name}_{vectorizer_name}".replace(" ", "_").lower()
    base = MODELS_DIR / stem

    if hasattr(model, "save") and callable(model.save):
        path = base.with_suffix(".pt")
        model.save(path)
    else:
        path = base.with_suffix(".pkl")
        with open(path, "wb") as fh:
            pickle.dump(model, fh)

    print(f"  Saved → {path}")
    return path

# Main pipeline
def run_ml_pipeline(
    cache_dir: str = "cache",
    labels: Optional[List[str]] = None,
    skip_mentalbert_lr: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Train every (vectorizer, model) pair, save all models, and write a
    joint CSV of evaluation metrics.

    :param cache_dir: Path to the cache directory with embeddings.
    :type cache_dir: str
    :param labels: Ordered list of class label strings.
    :type labels: list[str], optional
    :param skip_mentalbert_lr: Whether to skip Logistic Regression on MentalBERT.
    :type skip_mentalbert_lr: bool
    :returns: Combined results DataFrame, or None if nothing ran.
    :rtype: pandas.DataFrame or None
    """
    if labels is None:
        labels = ["Anxiety", "Depression", "Suicidal", "Normal"]

    n_classes = len(labels)
    RESULTS_DIR.mkdir(exist_ok=True)

    cache       = load_cache(cache_dir)
    y_train_raw = cache["y_train"]
    y_test_raw  = cache["y_test"]

    # Integer-encoded versions for the Torch NB models
    y_train_int = _encode_labels(y_train_raw, labels)
    y_test_int  = _encode_labels(y_test_raw,  labels)

    vectorizer_sets = {"TF-IDF": cache["tfidf"]}
    # MentalBERT results are disabled while running TF-IDF only.
    # if "mentalbert" in cache:
    #     vectorizer_sets["MentalBERT"] = cache["mentalbert"]

    all_results: List[pd.DataFrame] = []

    for vec_name, (X_train, X_test) in vectorizer_sets.items():

        print("\n" + "=" * 60)
        print(f"Vectorizer: {vec_name}")
        print("=" * 60)

        is_sparse = vec_name == "TF-IDF"
        models    = make_models(vec_name, allow_mentalbert_lr=not skip_mentalbert_lr)

        for model_name, model in models.items():

            print(f"\n--- {model_name} | {vec_name} ---")

            try:
                is_torch_nb = isinstance(
                    model, (TorchMultinomialNaiveBayes, TorchGaussianNaiveBayes)
                )

                # Torch NB needs integer labels; sklearn models use strings
                y_tr = y_train_int if is_torch_nb else y_train_raw
                y_te = y_test_int  if is_torch_nb else y_test_raw

                # Train and persist
                train_model(model, X_train, y_tr, n_classes)
                save_model(model, model_name, vec_name)

                # Wrap Torch NB so evaluate_model() receives a .predict(X) object
                # that returns string labels (not integers)
                if is_torch_nb:
                    label_map = {i: lbl for i, lbl in enumerate(labels)}
                    eval_model = _TorchModelWrapper(
                        model, is_sparse=is_sparse, label_map=label_map
                    )
                else:
                    eval_model = model

                # All models now return string labels, so use original labels
                eval_labels = labels
                stem = (
                    f"{model_name}_{vec_name}"
                    .replace(" ", "_")
                    .lower()
                )

                # Train split
                train_results, y_train_pred = evaluate_model(
                    model=eval_model, X=X_train,
                    y_true=y_train_raw, labels=eval_labels,
                    model_name=model_name, vectorizer_name=vec_name, split="Train",
                    save_csv=str(RESULTS_DIR / f"{stem}_train.csv"),
                )
                save_confusion_matrix(
                    y_true=y_train_raw, y_pred=y_train_pred, labels=eval_labels,
                    model_name=model_name, vectorizer_name=vec_name, split="Train",
                    save_dir=str(RESULTS_DIR),
                )

                # Test split
                test_results, y_test_pred = evaluate_model(
                    model=eval_model, X=X_test,
                    y_true=y_test_raw, labels=eval_labels,
                    model_name=model_name, vectorizer_name=vec_name, split="Test",
                    save_csv=str(RESULTS_DIR / f"{stem}_test.csv"),
                )
                save_confusion_matrix(
                    y_true=y_test_raw, y_pred=y_test_pred, labels=eval_labels,
                    model_name=model_name, vectorizer_name=vec_name, split="Test",
                    save_dir=str(RESULTS_DIR),
                )

                all_results.append(train_results)
                all_results.append(test_results)

            except Exception as exc:
                print(f"\n[ERROR] {model_name} | {vec_name} failed: {exc}")
                import traceback
                traceback.print_exc()

    # Joint CSV
    if all_results:
        summary_df   = pd.concat(all_results, ignore_index=True)
        summary_path = RESULTS_DIR / "final_model_results.csv"
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(summary_df.to_string(index=False))
        print(f"\nResults saved → {summary_path}")
        return summary_df

    print("No results generated.")
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train ML models and save evaluation tables to CSV."
    )
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="Directory with X_train/X_test caches.",
    )
    parser.add_argument(
        "--skip-mentalbert-lr",
        action="store_true",
        help="Skip Logistic Regression on MentalBERT to reduce CPU/RAM load.",
    )

    args = parser.parse_args()
    run_ml_pipeline(
        cache_dir=args.cache_dir,
        skip_mentalbert_lr=args.skip_mentalbert_lr,
    )
