#!/usr/bin/env python
# coding: utf-8

# # Mental Health Text Classification with SVM
# ### Vectorizers: CamemBERT (HuggingFace) + TF-IDF | Classifier: SVM (class-based) with GridSearch

# ## 1. Libraries

# In[1]:


# =========================
# Import Libraries
# =========================
import pandas as pd
import numpy as np

# Sklearn
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

# PyTorch + HuggingFace
import torch
from transformers import CamembertTokenizer, CamembertModel
from tqdm import tqdm

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# ## 2. Load & Clean Dataset

# In[2]:


# =========================
# Load and Clean Dataset
# =========================
data = pd.read_csv('C:\Users\Admin\Documents\FYP\french dataset\Code\MyResults\french_cleaned.csv')

# Remove rows with missing text or labels
data = data.dropna(subset=['cleaned_text', 'status'])
data['cleaned_text'] = data['cleaned_text'].astype(str)

print(f"Dataset shape: {data.shape}")
print(f"Label distribution:\n{data['status'].value_counts()}")
data.head()

# ## 3. Encode Labels

# In[ ]:


# =========================
# Encode Labels
# =========================
label_encoder = LabelEncoder()
data['encoded_label'] = label_encoder.fit_transform(data['status'])

print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
data[['status', 'encoded_label']].head()

# ## 4. Train-Test Split

# In[ ]:


# =========================
# Train-Test Split
# =========================
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    data['cleaned_text'],
    data['encoded_label'],
    test_size=0.2,
    random_state=42,
    stratify=data['encoded_label']
)

print(f"Train size: {len(X_train_texts)} | Test size: {len(X_test_texts)}")

# ## 5. CamemBERT Vectorizer
# > CamemBERT is a French RoBERTa-based model. Used here as a **feature extractor** (not fine-tuned). It produces a 768-dim CLS embedding per sentence.

# In[ ]:


# =========================
# CamemBERT Vectorizer (NOT a class — vectorizers stay functional)
# =========================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load CamemBERT tokenizer and model
camembert_tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
camembert_model = CamembertModel.from_pretrained('camembert-base')
camembert_model.to(device)
camembert_model.eval()


def get_camembert_embeddings(texts, tokenizer, model, device, batch_size=16):
    """
    Extract CLS-token embeddings from CamemBERT for a list of texts.
    Uses batching for efficiency.

    Args:
        texts      : list of strings
        tokenizer  : CamembertTokenizer
        model      : CamembertModel
        device     : torch.device
        batch_size : int, number of sentences per batch

    Returns:
        np.ndarray of shape (n_samples, 768)
    """
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="CamemBERT embedding"):
        batch = texts[i : i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # CLS token: first token of last hidden state
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)

# In[ ]:


# =========================
# Generate CamemBERT Embeddings
# =========================
X_train_cam = get_camembert_embeddings(
    X_train_texts.tolist(), camembert_tokenizer, camembert_model, device
)
X_test_cam = get_camembert_embeddings(
    X_test_texts.tolist(), camembert_tokenizer, camembert_model, device
)

print(f"Train embeddings shape: {X_train_cam.shape}")
print(f"Test  embeddings shape: {X_test_cam.shape}")

# ## 6. TF-IDF Vectorizer

# In[ ]:


# =========================
# TF-IDF Vectorizer (NOT a class — vectorizers stay functional)
# =========================
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_texts)
X_test_tfidf  = tfidf_vectorizer.transform(X_test_texts)

print(f"TF-IDF train shape: {X_train_tfidf.shape}")
print(f"TF-IDF test  shape: {X_test_tfidf.shape}")

# ## 7. SVM Model Class with GridSearch

# In[ ]:


# =========================
# SVM Classifier Class
# =========================

class SVMClassifier:
    """
    Support Vector Machine classifier with built-in GridSearch hyperparameter tuning.

    Attributes:
        best_model   : best SVC found by GridSearchCV
        best_params  : dict of best hyperparameters
        cv_results   : full GridSearchCV results dataframe
    """

    # Hyperparameter grid — covers kernel, C (regularization), gamma (for rbf/poly)
    PARAM_GRID = {
        'kernel' : ['rbf', 'poly', 'sigmoid'],
        'C'      : [0.1, 1, 10, 100],
        'gamma'  : ['scale', 'auto'],
    }

    def __init__(self, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1):
        """
        Args:
            cv      : number of cross-validation folds for GridSearch
            scoring : metric to optimise during grid search
            n_jobs  : parallel jobs (-1 = all cores)
            verbose : verbosity level of GridSearchCV
        """
        self.cv      = cv
        self.scoring = scoring
        self.n_jobs  = n_jobs
        self.verbose = verbose

        self.best_model  = None
        self.best_params = None
        self.cv_results  = None

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train):
        """
        Run GridSearchCV over PARAM_GRID and store the best estimator.

        Args:
            X_train : feature matrix (dense or sparse)
            y_train : target labels
        """
        grid_search = GridSearchCV(
            estimator  = SVC(),
            param_grid = self.PARAM_GRID,
            cv         = self.cv,
            scoring    = self.scoring,
            n_jobs     = self.n_jobs,
            verbose    = self.verbose,
            refit      = True   # refit best model on full train set
        )
        grid_search.fit(X_train, y_train)

        self.best_model  = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results  = pd.DataFrame(grid_search.cv_results_)

        print(f"\nBest params  : {self.best_params}")
        print(f"Best CV score ({self.scoring}): {grid_search.best_score_:.4f}")
        return self

    # ------------------------------------------------------------------
    def predict(self, X):
        """
        Predict class labels.

        Args:
            X : feature matrix
        Returns:
            np.ndarray of predicted labels
        """
        if self.best_model is None:
            raise RuntimeError("Call .fit() before .predict()")
        return self.best_model.predict(X)

    # ------------------------------------------------------------------
    def evaluate(self, X_test, y_test, label_encoder=None):
        """
        Compute and print evaluation metrics on the test set.
        *** Metrics placeholder — replace with your own metrics later ***

        Args:
            X_test        : test feature matrix
            y_test        : true labels
            label_encoder : sklearn LabelEncoder (for class names in report)

        Returns:
            dict with metric values
        """
        y_pred = self.predict(X_test)
        target_names = label_encoder.classes_ if label_encoder else None

        # ---------- METRICS PLACEHOLDER ----------
        metrics = {
            'accuracy'  : accuracy_score(y_test, y_pred),
            'f1_macro'  : f1_score(y_test, y_pred, average='macro'),
            # TODO: replace / add your metrics here
        }
        # -----------------------------------------

        print("\n--- Evaluation Results ---")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

        return metrics, y_pred

    # ------------------------------------------------------------------
    def plot_confusion_matrix(self, y_test, y_pred, label_encoder=None):
        """
        Plot a heatmap confusion matrix.

        Args:
            y_test        : true labels
            y_pred        : predicted labels
            label_encoder : sklearn LabelEncoder (for axis labels)
        """
        cm = confusion_matrix(y_test, y_pred)
        labels = label_encoder.classes_ if label_encoder else None

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d',
            xticklabels=labels,
            yticklabels=labels,
            cmap='Blues'
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix — SVM')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    def top_grid_results(self, n=10):
        """Return top-n grid search results sorted by mean test score."""
        cols = ['param_kernel', 'param_C', 'param_gamma',
                'mean_test_score', 'std_test_score', 'rank_test_score']
        return (
            self.cv_results[cols]
            .sort_values('rank_test_score')
            .head(n)
            .reset_index(drop=True)
        )

# ## 8. SVM + CamemBERT Embeddings

# In[ ]:


# =========================
# Train & Evaluate: SVM + CamemBERT
# =========================
print("="*50)
print("SVM with CamemBERT Embeddings")
print("="*50)

svm_cam = SVMClassifier(cv=5, scoring='f1_macro')
svm_cam.fit(X_train_cam, y_train)

metrics_cam, y_pred_cam = svm_cam.evaluate(X_test_cam, y_test, label_encoder)
svm_cam.plot_confusion_matrix(y_test, y_pred_cam, label_encoder)

print("\nTop 5 Grid Search Results:")
svm_cam.top_grid_results(5)

# ## 9. SVM + TF-IDF

# In[ ]:


# =========================
# Train & Evaluate: SVM + TF-IDF
# =========================
print("="*50)
print("SVM with TF-IDF")
print("="*50)

svm_tfidf = SVMClassifier(cv=5, scoring='f1_macro')
svm_tfidf.fit(X_train_tfidf, y_train)

metrics_tfidf, y_pred_tfidf = svm_tfidf.evaluate(X_test_tfidf, y_test, label_encoder)
svm_tfidf.plot_confusion_matrix(y_test, y_pred_tfidf, label_encoder)

print("\nTop 5 Grid Search Results:")
svm_tfidf.top_grid_results(5)

# ## 10. Cross-Validation (TF-IDF + Best SVM Params)

# In[ ]:


# =========================
# Cross-Validation on full dataset with best params from TF-IDF SVM
# =========================
tfidf_full = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_full     = tfidf_full.fit_transform(data['cleaned_text'])
y_full     = data['encoded_label']

# Use best params found during GridSearch
best_svm_cv = SVC(**svm_tfidf.best_params)

cv_results = cross_validate(
    best_svm_cv, X_full, y_full,
    cv=5,
    scoring=['accuracy', 'f1_macro'],
    n_jobs=-1
)

print(f"Cross-Validation Results (5-fold, TF-IDF + Best SVM)")
print(f"  Mean Accuracy : {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
print(f"  Mean F1 Macro : {cv_results['test_f1_macro'].mean():.4f} ± {cv_results['test_f1_macro'].std():.4f}")

# ## 11. Model Comparison Summary

# In[ ]:


# =========================
# Summary Comparison Table
# =========================
summary = pd.DataFrame([
    {
        'Vectorizer'  : 'CamemBERT',
        'Best Kernel' : svm_cam.best_params['kernel'],
        'Best C'      : svm_cam.best_params['C'],
        'Best Gamma'  : svm_cam.best_params['gamma'],
        'Accuracy'    : metrics_cam['accuracy'],
        'F1 Macro'    : metrics_cam['f1_macro'],
    },
    {
        'Vectorizer'  : 'TF-IDF',
        'Best Kernel' : svm_tfidf.best_params['kernel'],
        'Best C'      : svm_tfidf.best_params['C'],
        'Best Gamma'  : svm_tfidf.best_params['gamma'],
        'Accuracy'    : metrics_tfidf['accuracy'],
        'F1 Macro'    : metrics_tfidf['f1_macro'],
    }
])

print(summary.to_string(index=False))
summary
