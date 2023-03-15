from pytest import fixture

from matplotlib.figure import Figure
from numpy.random import seed as default_rng
from pycaret.datasets import get_data

from shap_clustering.model import ModelSelection

# set up fixed global random seed for reproducibility
default_rng(14032023)


@fixture(scope="module")
def df():
    return get_data("house")


@fixture(scope="module")
def selection(df):
    target = "SalePrice"
    selection = ModelSelection()
    selection.fit(df.sample(100), target)
    return selection


def test_ModelSelection_is_trained(selection):
    assert selection.trained_models is not None
    assert len(selection.trained_models) == 3
    assert selection.trained_models[0].__class__.__name__ == "GradientBoostingRegressor"


def test_ModelSelection_has_interpretation(selection):
    # Check that the interpretation is a dictionary
    # Check that the elements in interpretation are matplotlib figures
    interpretations = selection.interpret_models()
    assert isinstance(interpretations, dict)
    assert isinstance(interpretations["GradientBoostingRegressor"], Figure)
