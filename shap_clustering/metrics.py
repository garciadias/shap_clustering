from typing import Dict

import pandas as pd
from numpy import ndarray
from sklearn import metrics
from sklearn.base import ClassifierMixin, RegressorMixin

REGRESSION_METRICS = {
    "r2": metrics.r2_score,
    "mae": metrics.mean_absolute_error,
    "mse": metrics.mean_squared_error,
    "median_absolute_error": metrics.median_absolute_error,
    "explained_variance": metrics.explained_variance_score,
    "max_error": metrics.max_error,
}

CLASSIFICATION_METRICS = {
    "accuracy": metrics.accuracy_score,
    "balanced_accuracy": metrics.balanced_accuracy_score,
}


def metric_selector(model: object) -> dict:
    """Select the appropriate metrics for a given model.

    Parameters
    ----------
    model : object
        An instance of an scikit-learn model.

    Returns
    -------
    dict
        A dictionary of metrics for the model.
    """
    model_type = get_model_type(model)
    if model_type == "regressor":
        return REGRESSION_METRICS
    return CLASSIFICATION_METRICS


def get_model_type(model):
    """Determine whether an scikit-learn model is a classifier or a regressor.

    Parameters
    ----------
        model : object
          An instance of an scikit-learn model.

    Returns
    -------
    model_type : str
        "classifier" if the model is a classifier, "regressor" if it is a regressor.
    """
    if isinstance(model, RegressorMixin):
        return "regressor"
    if isinstance(model, ClassifierMixin):
        return "classifier"
    raise ValueError("The model must be an instance of an scikit-learn like regressor or classifier.")


def get_metrics(models: list, x_test: ndarray, y_test: ndarray, sort_by=None) -> pd.DataFrame:
    """Return metrics for the trained models.

    Parameters
    ----------
    models : list
        A list of trained models.
    x_test : ndarray
        The test set features.
    y_test : ndarray
        The test set target.
    sort_by : str, optional
        The metric to sort the results by, by default None

    Returns
    -------
    pd.DataFrame
        A DataFrame of metrics for the trained models.

    Raises
    ------
    ValueError
        If the model is not an instance of an scikit-learn like regressor or classifier.
    """
    metrics_dict: Dict[str, Dict[str, float]] = {}
    for model in models:
        model_metric: Dict[str, float] = {}
        y_pred = model.predict(x_test)
        metrics_list = metric_selector(model)
        for metric_name, metric in metrics_list.items():
            model_metric[metric_name] = metric(y_test, y_pred)
        model_name = model.__class__.__name__
        metrics_dict[model_name] = model_metric
    metrics = pd.DataFrame.from_dict(metrics_dict).T
    if (sort_by is None) or (sort_by not in metrics.columns):
        sort_by = metrics.columns[0]
    metrics = metrics.sort_values(sort_by, ascending=False)
    return metrics
