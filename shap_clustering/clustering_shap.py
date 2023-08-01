from typing import Optional

import pandas as pd
from sklearn.mixture import GaussianMixture


def run_gmm(shap_values: pd.DataFrame, n_components: Optional[int] = None) -> GaussianMixture:
    """Run Gaussian Mixture Model on the data.

    Parameters
    ----------
    shap_values : pd.DataFrame
        Dataframe of features.
    n_components : int, optional
        Number of components, by default 3

    Returns
    -------
    GaussianMixture
        Fitted Gaussian Mixture Model.
    """
    if n_components is None:
        n_components = find_best_number_of_components(shap_values)
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(shap_values)
    return gmm


def find_best_number_of_components(shap_values: pd.DataFrame, max_components: int = 10) -> Optional[int]:
    """Find the best number of components for the Gaussian Mixture Model.

    Parameters
    ----------
    shap_values : pd.DataFrame
        Dataframe of features.
    max_components : int, optional
        Maximum number of components to be tested, by default 10

    Returns
    -------
    int
        Best number of components.
    """
    best_bic = float("inf")
    best_n_components = None
    for n_components in range(1, max_components + 1):
        gmm = run_gmm(shap_values, n_components=n_components)
        bic = gmm.bic(shap_values)
        if bic < best_bic:
            best_bic = bic
            best_n_components = n_components
    return best_n_components
