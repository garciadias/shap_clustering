from matplotlib.figure import Figure
from numpy.random import seed as default_rng
from pytest import fixture
from sklearn.datasets import fetch_california_housing
from sklearn.utils.validation import check_is_fitted

from shap_clustering.model import ModelSelection

# set up fixed global random seed for reproducibility
default_rng(14032023)


@fixture(scope="module")
def df():
    return fetch_california_housing(as_frame=True).frame.sample(100)


@fixture(scope="module")
def selection(df):
    target = "MedHouseVal"
    selection = ModelSelection()
    selection.fit(df, target)
    return selection


def test_ModelSelection_is_trained(selection):
    assert selection.models is not None
    assert len(selection.models) == 3
    assert selection.models[0].__class__.__name__ == "LinearRegression"
    assert all([check_is_fitted(model) is None for model in selection.models])


def test_ModelSelection_has_pdp(selection, df):
    # Check that the interpretation is a dictionary
    # Check that the elements in interpretation are matplotlib figures
    first_var = df.columns[0]
    explanation = selection.explain()
    assert isinstance(explanation.pdp, dict)
    assert isinstance(explanation.pdp[first_var]["LinearRegression"], Figure)
