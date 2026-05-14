"""
cache_embeddings.py

Usage
-----
    python cache_embeddings.py
"""

import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import save_npz
import pandas as pd
from sklearn.model_selection import train_test_split
from vectorizers import build_tfidf_features, generate_mentalbert_embeddings


def cache_embeddings(
    train_texts: list,
    test_texts: list,
    y_train,
    y_test,
    cache_dir: str = "cache",
    use_mentalbert: bool = True,
) -> None:
    """
    Vectorize train/test texts and save results to disk.

    Parameters
    ----------
    train_texts : list[str]
        Cleaned training texts.
    test_texts : list[str]
        Cleaned test texts.
    y_train : array-like
        Training labels.
    y_test : array-like
        Test labels.
    cache_dir : str
        Directory to save all cached files.
    use_mentalbert : bool
        Set False to skip MentalBERT (slow on CPU).
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Save labels (shared by all mains)
    joblib.dump(y_train, cache_path / "y_train.pkl")
    joblib.dump(y_test,  cache_path / "y_test.pkl")

    # ── TF-IDF ────────────────────────────────────────────────────────
    print("Vectorizing with TF-IDF...")
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = build_tfidf_features(
        train_texts, test_texts
    )
    # sparse matrices use save_npz
    save_npz(str(cache_path / "X_train_tfidf.npz"), X_train_tfidf)
    save_npz(str(cache_path / "X_test_tfidf.npz"),  X_test_tfidf)
    # save the fitted vectorizer so inference can reuse the same vocab
    joblib.dump(tfidf_vectorizer, cache_path / "tfidf_vectorizer.pkl")
    print(f"  TF-IDF saved → {cache_path}/X_{{train,test}}_tfidf.npz")

    # ── MentalBERT ────────────────────────────────────────────────────
    if use_mentalbert:
        print("Vectorizing with MentalBERT (this may take a while)...")
        X_train_mb = generate_mentalbert_embeddings(list(train_texts))
        X_test_mb  = generate_mentalbert_embeddings(list(test_texts))
        np.save(str(cache_path / "X_train_mentalbert.npy"), X_train_mb)
        np.save(str(cache_path / "X_test_mentalbert.npy"),  X_test_mb)
        print(f"  MentalBERT saved → {cache_path}/X_{{train,test}}_mentalbert.npy")

    print(f"\nAll embeddings cached in '{cache_dir}/'.")


if __name__ == "__main__":
    # Load the already-cleaned dataset
    df = pd.read_csv("/home/joseph_am_makhlouf/my_projects/english_mh/plots/processed_data_with_features.csv")

    df = df.dropna(subset=["cleaned_text"])
    df["cleaned_text"] = df["cleaned_text"].astype(str)

    X = df["cleaned_text"].tolist()
    y = df["status"].tolist()

    train_texts, test_texts, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    cache_embeddings(
        train_texts=train_texts,
        test_texts=test_texts,
        y_train=y_train,
        y_test=y_test,
        cache_dir="cache",
        use_mentalbert=True,
    )