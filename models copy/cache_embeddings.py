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
    train_texts_raw: list,   # raw for MentalBERT
    test_texts_raw: list,
    cache_dir: str = "cache",
    use_mentalbert: bool = True,
) -> None:
    """
    Vectorize train/test texts and save results to disk.

    :param train_texts: Cleaned training texts.
    :type train_texts: list[str]
    :param test_texts: Cleaned test texts.
    :type test_texts: list[str]
    :param y_train: Training labels.
    :type y_train: array-like
    :param y_test: Test labels.
    :type y_test: array-like
    :param train_texts_raw: Raw training texts used for MentalBERT embeddings.
    :type train_texts_raw: list[str]
    :param test_texts_raw: Raw test texts used for MentalBERT embeddings.
    :type test_texts_raw: list[str]
    :param cache_dir: Directory to save all cached files.
    :type cache_dir: str
    :param use_mentalbert: Whether to compute MentalBERT embeddings.
    :type use_mentalbert: bool
    :returns: None
    :rtype: None
    """
    # Create the target cache directory if it does not exist.
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Save labels (shared by all mains)
    joblib.dump(y_train, cache_path / "y_train.pkl")
    joblib.dump(y_test,  cache_path / "y_test.pkl")

    # TF-IDF
    print("Vectorizing with TF-IDF...")
    # Build TF-IDF features from cleaned text for both train and test splits.
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = build_tfidf_features(
        train_texts, test_texts
    )
    # sparse matrices use save_npz
    save_npz(str(cache_path / "X_train_tfidf.npz"), X_train_tfidf)
    save_npz(str(cache_path / "X_test_tfidf.npz"),  X_test_tfidf)
    # save the fitted vectorizer so inference can reuse the same vocabulary
    joblib.dump(tfidf_vectorizer, cache_path / "tfidf_vectorizer.pkl")
    print(f"  TF-IDF saved → {cache_path}/X_{{train,test}}_tfidf.npz")

    # MentalBERT embeddings are computed from the raw text column only.
    if use_mentalbert:
        print("Vectorizing with MentalBERT (this may take a while)...")
        # Generate raw-text sentence embeddings using the MentalBERT transformer.
        X_train_mb = generate_mentalbert_embeddings(list(train_texts_raw))
        X_test_mb  = generate_mentalbert_embeddings(list(test_texts_raw))

        np.save(str(cache_path / "X_train_mentalbert.npy"), X_train_mb)
        np.save(str(cache_path / "X_test_mentalbert.npy"),  X_test_mb)
        print(f"  MentalBERT saved to   {cache_path}/X_{{train,test}}_mentalbert.npy")

    print(f"\nAll embeddings cached in '{cache_dir}/'.")


if __name__ == "__main__":
    # Load the already-cleaned dataset
    df = pd.read_csv("/home/joseph_am_makhlouf/my_projects/english_mh/plots/processed_data_with_features.csv")
    df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # Drop rows where cleaned_text is empty after fillna
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)


    # Split the dataframe
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["status"], random_state=42
    )

    train_texts = train_df["cleaned_text"].tolist()
    test_texts = test_df["cleaned_text"].tolist()
    train_texts_raw = train_df["text"].tolist()
    test_texts_raw = test_df["text"].tolist()
    y_train = train_df["status"].tolist()
    y_test = test_df["status"].tolist()

    cache_embeddings(
        train_texts=train_texts,
        test_texts=test_texts,
        y_train=y_train,
        y_test=y_test,
        train_texts_raw=train_texts_raw,
        test_texts_raw=test_texts_raw,
        cache_dir="cache",
        use_mentalbert=True,
    )