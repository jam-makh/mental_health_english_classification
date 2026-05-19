"""
SVM classifier for mental health text classification.
"""

import numpy as np
import pandas as pd
import warnings

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")


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
        "kernel": ["rbf", "poly", "sigmoid"],
        "C":      [0.1, 1, 10, 100],
        "gamma":  ["scale", "auto"],
    }

    def __init__(self, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1):
        self.cv      = cv
        self.scoring = scoring
        self.n_jobs  = n_jobs
        self.verbose = verbose
        self.model       = None  # main_ml.py calls .model
        self.best_params = None
        self.cv_results  = None

    # fit changed to train to match my code
    def train(self, X_train, y_train):
        """
        Run GridSearchCV over PARAM_GRID and store the best estimator.

        Args:
            X_train : feature matrix (dense or sparse)
            y_train : target labels
        """
        grid_search = GridSearchCV(
            estimator=SVC(),
            param_grid=self.PARAM_GRID,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            refit=True,    # refit best model on full train set
        )
        grid_search.fit(X_train, y_train)
        self.model       = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results  = pd.DataFrame(grid_search.cv_results_)
        print(f"Best params: {self.best_params}")
        print(f"Best CV {self.scoring}: {grid_search.best_score_:.4f}")

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X : feature matrix
        Returns:
            np.ndarray of predicted labels
        """
        if self.model is None:
            raise RuntimeError("Call .train() before .predict()")
        return self.model.predict(X)
