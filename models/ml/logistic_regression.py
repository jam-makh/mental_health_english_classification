import numpy as np

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_validate
)
from typing import  Dict, Any, Optional, List


class LogisticRegressionModel:
    """
    Logistic Regression class that handles train/test split,
    hyperparameters tuning, model fitting, predicition
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

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> List[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the dataset into train and test sets.
        
        :param X: Feature matrix.
        :type X: np.ndarray
        :param y: Target labels.
        :type y: np.ndarray
        :returns: Split datasets.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )

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
        param_grid = {
            "penalty": ["l2", "l1", "elasticnet"],
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs", "saga", "liblinear"],
            "class_weight": [None, "balanced"]
        }

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
            return_train_score=True
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

    def cross_validate_model(
        self,
        X,
        y,
        cv_splits: int = 5
    ) -> Dict:
        """
        Perform cross-validation using the best model.

        :param X: Feature matrix.
        :type X: array-like
        :param y: Target labels.
        :type y: array-like
        :param cv_splits: Number of cross-validation folds, by default 5.
        :type cv_splits: int, optional
        :returns: Cross-validation metrics.
        :rtype: Dict
        :param cv_splits: Number of cross-validation folds, by default 5.
        :type cv_splits: int, optional
        """

        if self.model is None:
            raise ValueError("Train the model before cross-validation.")

        scoring = [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "support"
        ]

        cv_results = cross_validate(
            self.model,
            X,
            y,
            cv=cv_splits,
            scoring=scoring,
            n_jobs=-1
        )

        return {
            "accuracy_mean": np.mean(cv_results["test_accuracy"]),
            "precision_macro_mean": np.mean(
                cv_results["test_precision_macro"]
            ),
            "recall_macro_mean": np.mean(
                cv_results["test_recall_macro"]
            ),
            "f1_macro_mean": np.mean(
                cv_results["test_f1_macro"]
            ),
            "support_mean": np.mean(
                cv_results["test_support"]
            )
        }
