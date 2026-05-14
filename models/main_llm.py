"""
main_llm.py

LLMs take raw text directly — no vectorizer, no cache needed.
Just iterate over all LLM models and evaluate.

Run order
---------
    1. python main_llm.py   (independent of cache_embeddings.py)
"""

import pandas as pd

from models.evaluation import evaluate_model
from models.llm.gpt    import GPTModel
from models.llm.llama  import LlamaModel
from models.llm.gemini import GeminiModel


def run_llm_pipeline(
    test_texts: list,
    y_test,
    labels: list = None,
) -> pd.DataFrame:
    """
    Run every LLM model on raw text and evaluate.

    LLMs do their own tokenization internally — no vectorizer needed.
    If a model also requires fine-tuning, add model.train(train_texts, y_train)
    before predict().

    Parameters
    ----------
    test_texts : list[str]
        Cleaned test texts.
    y_test : array-like
        Test labels.
    labels : list[str], optional
        Class label names.

    Returns
    -------
    pd.DataFrame
        One row per model with all metrics.
    """
    if labels is None:
        labels = ["Anxiety", "Depression", "Suicidal", "Normal"]

    models = {
        "GPT":    GPTModel(),
        "LLaMA":  LlamaModel(),
        "Gemini": GeminiModel(),
    }

    all_results = []

    for model_name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"Running {model_name}...")
        print("=" * 60)

        y_pred = model.predict(test_texts)

        result = evaluate_model(
            y_true=y_test,
            y_pred=y_pred,
            labels=labels,
            model_name=model_name,
            show_confusion_matrix=True,
        )

        macro = result["classification_report"]["macro avg"]
        all_results.append({
            "Model":           model_name,
            "Vectorizer":      "None (raw text)",
            "Accuracy":        round(result["accuracy"], 4),
            "Macro Precision": round(macro["precision"], 4),
            "Macro Recall":    round(macro["recall"], 4),
            "Macro F1":        round(macro["f1-score"], 4),
            "Total Support":   int(macro["support"]),
        })

    summary_df = pd.DataFrame(all_results)
    print("\n" + "=" * 60)
    print("LLM RESULTS SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    return summary_df


if __name__ == "__main__":
    from your_data_loader import load_data
    _, test_texts, _, y_test = load_data()

    summary = run_llm_pipeline(test_texts=test_texts, y_test=y_test)
    summary.to_csv("llm_results.csv", index=False)
