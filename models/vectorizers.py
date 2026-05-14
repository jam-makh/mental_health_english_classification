"""
TF-IDF vectorization utilities.
"""

from typing import Tuple, List
from tqdm import tqdm

import numpy as np
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel


def build_tfidf_features(
    train_texts,
    test_texts,
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2)
):
    """
    Convert text into TF-IDF vectors.

    Parameters
    ----------
    train_texts : list
        Training text samples.

    test_texts : list
        Test text samples.

    max_features : int, optional
        Maximum vocabulary size.

    ngram_range : Tuple[int, int], optional
        N-gram range.

    Returns
    -------
    Tuple
        X_train, X_test, fitted_vectorizer
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )

    # Learn vocabulary from training data
    X_train = vectorizer.fit_transform(train_texts)

    # Transform test data using same vocabulary
    X_test = vectorizer.transform(test_texts)

    return X_train, X_test, vectorizer

def mean_pooling(
    model_output,
    attention_mask
):
    """
    Apply mean pooling to transformer outputs.
    """

    token_embeddings = model_output[0]

    input_mask_expanded = (
        attention_mask.unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    return torch.sum(
        token_embeddings * input_mask_expanded,
        1
    ) / torch.clamp(
        input_mask_expanded.sum(1),
        min=1e-9
    )


def generate_mentalbert_embeddings(
    texts: List[str],
    batch_size: int = 16,
    max_length: int = 256,
    model_name: str = "mental/mental-bert-base-uncased"
) -> np.ndarray:
    """
    Generate MentalBERT sentence embeddings.

    Parameters
    ----------
    texts : List[str]
        Input text samples.

    batch_size : int, optional
        Batch size.

    max_length : int, optional
        Maximum token length.

    Returns
    -------
    np.ndarray
        Dense embedding matrix.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model.eval()

    embeddings = []

    with torch.no_grad():

        for i in tqdm(range(0, len(texts), batch_size), desc="  Generating MentalBERT embeddings", unit="batch"):

            batch = texts[i:i + batch_size]

            encoded_input = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            model_output = model(**encoded_input)

            sentence_embeddings = mean_pooling(
                model_output,
                encoded_input["attention_mask"]
            )

            embeddings.append(
                sentence_embeddings.cpu().numpy()
            )

    return np.vstack(embeddings)
