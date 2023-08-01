import pytest

from shap_clustering.metrics import metric_selector


def test_metric_selector_raises_correct_error():
    error_message = "The model must be an instance of an scikit-learn like regressor or classifier."
    with pytest.raises(ValueError) as error:
        metric_selector("test")
    assert str(error.value) == error_message
