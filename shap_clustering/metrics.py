import pandas as pd
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

CLASSIFICATION_METRICS = {"report": metrics.classification_report}


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
    elif model_type == "classifier":
        return CLASSIFICATION_METRICS
    else:
        raise ValueError("The model must be an instance of an scikit-learn regressor or classifier.")


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
    elif isinstance(model, ClassifierMixin):
        return "classifier"
    else:
        raise ValueError("The model must be an instance of an scikit-learn regressor or classifier.")


def get_metrics(models: object, X_test, y_test) -> pd.DataFrame:
    """Return metrics for the trained models.

    Returns
    -------
    pd.DataFrame : dataframe
        Dataframe of metrics for each trained model.

    """
    metrics = {}
    for model in models:
        model_name = model.__class__.__name__
        metrics[model_name] = {}
        y_pred = model.predict(X_test)
        metrics_list = metric_selector(model)
        for metric_name, metric in metrics_list.items():
            metrics[model_name][metric_name] = metric(y_test, y_pred)
    metrics = pd.DataFrame.from_dict(metrics).T
    if "r2" in metrics.columns:
        metrics = metrics.sort_values("r2", ascending=False)
    return metrics
