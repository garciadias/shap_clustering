from pandas import DataFrame
from matplotlib.figure import Figure
from numpy import ndarray
from numpy.random import seed as default_rng
from pytest import fixture
from shap import Explanation
from sklearn.datasets import load_diabetes
from sklearn.utils.validation import check_is_fitted

from shap_clustering.model import ModelSelection

# set up fixed global random seed for reproducibility
default_rng(14032023)


@fixture(scope="module")
def df():
    return load_diabetes(as_frame=True).frame.sample(100)


@fixture(scope="module")
def selection(df):
    target = "target"
    selection = ModelSelection()
    selection.fit(df, target)
    return selection


def test_ModelSelection_is_trained(selection):
    assert selection.models is not None
    assert len(selection.models) == 3
    assert selection.models[0].__class__.__name__ == "LinearRegression"
    assert all([check_is_fitted(model) is None for model in selection.models])


# def test_ModelSelection_has_pdp(selection, df):
#     # Check that the interpretation is a dictionary
#     # Check that the elements in interpretation are matplotlib figures
#     first_var = df.columns[0]
#     explanation = selection.explain()
#     assert isinstance(explanation.pdp, dict)
#     assert isinstance(explanation.pdp[first_var]["LinearRegression"], Figure)


def test_ModelSelection_has_shap_values(selection):
    # Check that the interpretation is a dictionary
    # Check that the elements in interpretation are matplotlib figures
    explanation = selection.explain()
    assert isinstance(explanation.shap_values_, dict)
    assert isinstance(explanation.shap_values_["LinearRegression"], ndarray)
    assert isinstance(explanation.shap_values_["LGBMRegressor"], ndarray)
    assert isinstance(explanation.shap_values_["ElasticNet"], ndarray)

def test_ModelSelection_has_shap_importance(selection):
    # Check that the interpretation is a dictionary
    # Check that the elements in interpretation are matplotlib figures
    explanation = selection.explain()
    assert isinstance(explanation.shap_importance_, DataFrame)
    assert explanation.shap_importance_.shape == (selection.X_train.shape[1], len(selection.models))

def test_ModelSelection_has_importace_plot(selection):
    # Check that the interpretation is a dictionary
    # Check that the elements in interpretation are matplotlib figures
    explanation = selection.explain()
    assert isinstance(explanation.importance_plot(), Figure)