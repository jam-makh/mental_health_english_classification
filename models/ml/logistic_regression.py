import numpy as np

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from typing import  Dict, Any, Optional, Tuple


class LogisticRegressionModel:
    """
    Logistic Regression class that handles train/test split,
    hyperparameters tuning, model fitting, predicition
    Receives pre-split, pre-vectorized arrays only.
    Splitting and vectorization are handled upstream
    in cache_embeddings.py to avoid recomputing
    expensive MentalBERT embeddings.
    """
    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.2
    ) -> None:

        self.random_state = random_state
        self.test_size = test_size

        self.model: Optional[LogisticRegression] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.grid_search: Optional[GridSearchCV] = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scoring: str = "f1_macro",
        cv_splits: int = 5
    ) -> None:
        """
        Train the Logistic Regression model using GridSearchCV.
        
        :param X_train: Training feature matrix.
        :type X_train: np.ndarray
        :param y_train: Training labels.
        :type y_train: np.ndarray
        :param scoring: Metric used for model selection, by default "f1_macro".
        :type scoring: str, optional
        :param cv_splits: Number of cross-validation folds, by default 5.
        :type cv_splits: int, optional
        :returns: None
        :rtype: None
        """

        # Define base Logistic Regression model
        base_model = LogisticRegression(
            max_iter=5000,
            random_state=self.random_state
        )

        # Hyperparameter search space
        param_grid =  [
            # lbfgs: l2 or None only
            {
                "solver":       ["lbfgs"],
                "penalty":      ["l2", None],
                "C":            [0.01, 0.1, 1.0, 10.0],
                # class_weighttells the model to penalize mistakes
                # on minority classes more heavily during training.
                "class_weight": [None, "balanced"],
            },
            # saga: all penalties including elasticnet
            {
                "solver":       ["saga"],
                "penalty":      ["l1", "l2", "elasticnet"],
                "C":            [0.01, 0.1, 1.0, 10.0],
                "l1_ratio":     [0.1, 0.5, 0.9],
                "class_weight": [None, "balanced"],
            },
            # liblinear: l1/l2 only, works for binary classification (french dataset)
            # skipped on multiclass
            {
                "solver":       ["liblinear"],
                "penalty":      ["l1", "l2"],
                "C":            [0.01, 0.1, 1.0, 10.0],
                "class_weight": [None, "balanced"],
            },
        ]

        # Stratified CV preserves label distribution
        stratified_cv = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=self.random_state
        )

        # GridSearchCV for hyperparameter optimization
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=scoring,
            cv=stratified_cv,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
            error_score = 0.0
        )

        # Train all parameter combinations
        print("Running GridSearchCV...")
        with tqdm(total=1, desc="Logistic Regression GridSearch") as pbar:
            grid_search.fit(X_train, y_train)
            pbar.update(1)
    
        # Store fitted GridSearchCV object
        self.grid_search_ = grid_search

        # Store best estimator
        self.model = grid_search.best_estimator_

        # Store best parameters
        self.best_params_ = grid_search.best_params_

    def predict(
        self,
        X_test: np.ndarray
    ) -> np.ndarray:
        """
        Predict labels using the trained model.

        :param X_test: Test feature matrix.
        :type X_test: np.ndarray
        :returns: Predicted labels.
        :rtype: np.ndarray
        """

        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        return self.model.predict(X_test)
