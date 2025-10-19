from typing import Optional, Any
import logging

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def trainer(
    dataset: Dataset,
    param_grid: Optional[dict[str, Any]] = None,
    max_iter: int = 5000,
    setup_mode: bool = False,
    save_model: bool = False,
    filename: str = "logreg.pkl"
) -> tuple[pd.DataFrame, float, dict]:
    """
    Perform logistic regression model selection using grid search and TF-IDF pipeline.

    Parameters
    ----------
    dataset : Dataset
        HuggingFace Dataset object with 'text' and 'label' columns.
    param_grid : dict[str, Any]
        Dictionary specifying parameter search space for GridSearchCV.
        If empty, uses the default grid.
    max_iter : int, optional
        Maximum number of iterations for solver (default is 5000).
    setup_mode : bool, optional
        If True, return detailed results and best estimator (default is False).
    save_model : bool, optional
        If True, saves the best estimator to the path_to_save (default is False).
    path_to_save : str, optional
        File path to save the trained model (default is "./src/best_model/logreg.pkl").

    Returns
    -------
    all_res : pandas.DataFrame
        DataFrame with mean test scores and parameters (if setup_mode=True).
    best_score : float
        Highest f1_macro score during grid search (if setup_mode=True).
    best_params : dict
        Best hyperparameters (if setup_mode=True).
    best_estimator : fitted estimator
        Best estimator object (if setup_mode=True).
    or
    best_estimator : fitted estimator
        Best estimator object (if setup_mode=False).

    Notes
    -----
    Uses scikit-learn Pipeline: TfidfVectorizer + LogisticRegression.
    Optimizes f1_macro via GridSearchCV.
    """
    X = list(dataset['text'])
    y = list(dataset['label'])

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000)),
        ('estimator', LogisticRegression(max_iter=max_iter))
    ])

    if not param_grid:
        param_grid = {
            "estimator__C": np.linspace(700, 1000, 10),
            "estimator__tol": [1e-1, 1e-2, 1e-3],
            "estimator__class_weight": ['balanced'],
            "estimator__solver": ['newton-cholesky'],
        }
        logger.info("Using default parameter grid")

    LR_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1_macro',
        n_jobs=8,
        return_train_score=True,
    )

    try:
        logger.info("Starting grid search...")
        LR_search.fit(X, y)
        logger.info("Grid search completed")
    except Exception as e:
        logger.error(f"Error during grid search: {e}")
        raise
    
    if setup_mode:
        all_res = (
            pd.DataFrame({
                "mean_test_score": LR_search.cv_results_["mean_test_score"],
                "std_test_score": LR_search.cv_results_["std_test_score"],
                "mean_train_score": LR_search.cv_results_["mean_train_score"],
                "mean_fit_time": LR_search.cv_results_["mean_fit_time"]
            })
            .join(pd.json_normalize(LR_search.cv_results_["params"]).add_prefix("param_"))
        )

    logger.info(f"Best f1_macro score: {LR_search.best_score_:.4f}")
    logger.info(f"Best parameters: {LR_search.best_params_}")

    if save_model:
        try:
            joblib.dump(LR_search.best_estimator_, filename)
            logger.info(f"Best model saved: {filename}")
        except Exception as e:
            logger.error(f"Error save process: {e}")
            raise

    if setup_mode:
        return all_res, LR_search.best_score_, LR_search.best_params_, LR_search.best_estimator_
    else:
        return LR_search.best_estimator_