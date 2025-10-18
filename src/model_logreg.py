from typing import Optional

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def trainer(
    dataset: Dataset,
    param_grid: Optional[dict[str, any]] = None,
) -> tuple[pd.DataFrame, float, dict]:
    """
    Perform logistic regression model selection using grid search and TF-IDF pipeline.

    Parameters
    ----------
    dataset : Dataset
        HuggingFace Dataset object with 'text' and 'label' columns.

    param_grid : dict[str, any]
        Dictionary specifying parameter search space for GridSearchCV.
        If empty, uses the default grid.

    Returns
    -------
    all_res : pandas.DataFrame
        Table with mean test scores, fit time and used parameters for each grid point.
    best_score : float
        The highest f1_macro score found during grid search.
    best_params : dict
        Dictionary of best hyperparameters.

    Notes
    -----
    Uses scikit-learn Pipeline: TfidfVectorizer + LogisticRegression.
    Optimizes f1_macro via GridSearchCV.
    """
    X = list(dataset['text'])
    y = list(dataset['label'])

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('estimator', LogisticRegression())
    ])
    if not param_grid:
        param_grid = {
            "estimator__C": np.linspace(0.001, 1000, 10),
            "estimator__tol": [1e-2, 1e-4, 1e-6],
            "estimator__class_weight": ['balanced'],
            "estimator__solver": ['lbfgs', 'saga', 'newton-cholesky'],
            "estimator__max_iter": [5000, 10000]
        }
    LR_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1_macro',
        n_jobs=8,
        return_train_score=True,
    )
    LR_search.fit(X, y)
    all_res = (
        pd.DataFrame({
            "mean_test_score": LR_search.cv_results_["mean_test_score"],
            "mean_fit_time": LR_search.cv_results_["mean_fit_time"]
        })
        .join(pd.json_normalize(LR_search.cv_results_["params"]).add_prefix("param_"))
    )
    return all_res, LR_search.best_score_, LR_search.best_params_