from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from warnings import warn

import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMRegressor
from matplotlib.figure import Figure
from shap import Explainer
from shap.utils._exceptions import ExplainerError
from shap.plots import partial_dependence
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from shap_clustering.clustering_shap import run_gmm
from shap_clustering.metrics import get_metrics


@dataclass
class Explanation:
    """Class for interpreting a selection of trained models."""

    model_selection: "ModelSelection"

    def __post_init__(self):
        self.shap_values_ = self._shap_values()
        self.shap_importance_ = self._shap_importance()

    def _shap_importance(self) -> pd.DataFrame:
        """Return shap importance for the trained models.

        Returns
        -------
        pd.DataFrame : dataframe
            Dataframe of SHAP importance for each trained model.

        """
        feature_columns = self.model_selection.X_train.columns
        shap_importance = []
        for shap_values in self.shap_values_.values():
            shap_values_train = shap_values[shap_values["train_test"].isin(["train"])][feature_columns]
            shap_importance.append(shap_values_train.abs().mean(axis=0))
        feature_indexes = list(self.shap_values_.keys())
        importances = pd.DataFrame(shap_importance, columns=feature_columns, index=feature_indexes).T
        sort_model = self.model_selection.metrics.iloc[0].name
        importances = importances.sort_values(sort_model, ascending=True)
        return importances

    def get_pdps(self) -> Dict[Any, Any]:
        """Interpret the trained models using Partial Dependence Plots.

        Returns
        -------
        dict : dictionary
            Dictionary of Partial Dependence Plots for each trained model.

        """
        pdp = {}
        for feature in self.model_selection.X_train.columns:
            feature_plots = {}
            for model in self.model_selection.models:
                model_name = model.__class__.__name__
                partial_dependence(
                    feature,
                    model.predict,
                    resample(self.model_selection.X_train),
                    model_expected_value=True,
                    ice=False,
                    show=False,
                )
                feature_plots[model_name] = plt.gca()
                plt.close()
            pdp[feature] = feature_plots
        return pdp

    def _shap_values(self) -> Dict:
        """Return shap values for the trained models.

        Returns
        -------
        dict : dictionary
            Dictionary of SHAP values for each trained model.

        """
        shap_values = {}
        for model in self.model_selection.models:
            # get shap values for train and test data
            explainer = Explainer(model, self.model_selection.X_train)
            shap_values_train = explainer.shap_values(self.model_selection.X_train)
            try:
                shap_values_test = explainer.shap_values(self.model_selection.X_test)
            except ExplainerError as e:
                model_name = model.__class__.__name__
                warn(f"Ignoring ExplainerError for test set on {model_name} by using `check_additivity=False`: {e}.")
                shap_values_test = explainer.shap_values(self.model_selection.X_test, check_additivity=False)
            # combine train and test shap values on a single DataFrame
            shap_values_train_df = pd.DataFrame(shap_values_train, columns=self.model_selection.X_train.columns)
            shap_values_train_df["train_test"] = ["train"] * shap_values_train_df.shape[0]
            shap_values_test_df = pd.DataFrame(shap_values_test, columns=self.model_selection.X_test.columns)
            shap_values_test_df["train_test"] = ["test"] * shap_values_test_df.shape[0]
            combined_shap_values = pd.concat([shap_values_train_df, shap_values_test_df])
            # add model name to the shap values
            model_name = model.__class__.__name__
            shap_values[model_name] = combined_shap_values
        return shap_values

    def importance_plot(self) -> Figure:
        """Plot SHAP importance for the trained models.

        Returns
        -------
        Figure : matplotlib figure
            Figure of SHAP importance for each trained model.

        """
        fig, ax = plt.subplots()
        normalized_importance = self.shap_importance_.div(self.shap_importance_.max(axis=0), axis=1)
        normalized_importance.plot.barh(ax=ax)
        return fig

    def cluster_shap_values(self, n_components: Optional[int] = None) -> pd.DataFrame:
        """Cluster the SHAP values for the trained models.

        Parameters
        ----------
        n_components : int, optional, default=5
            Number of clusters to be created.

        Returns
        -------
        pd.DataFrame : dataframe
            Dataframe of cluster assignation for the shap values using the clustering results from the best model
            shapley train values.
        """
        feature_columns = self.model_selection.X_train.columns
        self.gmm_models_ = {}
        for model_name, shap_values in self.shap_values_.items():
            shap_values_train = shap_values[shap_values["train_test"].isin(["train"])][feature_columns]
            self.gmm_models_[model_name] = run_gmm(shap_values_train, n_components=n_components)
        # create empty dataframe
        shap_values_cluster = pd.DataFrame(columns=self.shap_values_.keys(), index=shap_values_train.index)
        # fill dataframe with cluster assignation
        for model_name, shap_values in self.shap_values_.items():
            shap_values_cluster[model_name] = self.gmm_models_[model_name].predict(shap_values[feature_columns])
        return shap_values_cluster


@dataclass
class ModelSelection:
    """Class for selecting the best model from a list of models.

    Parameters
    ----------
    models : list, optional, default=[LinearRegression(), RandomForestRegressor(), lgb.LGBMRegressor()]
        List of models to be trained and evaluated.

    """

    models: list = field(
        default_factory=lambda: [
            LinearRegression(),
            ElasticNet(),
            LGBMRegressor(),
        ]
    )

    def fit(self, df: pd.DataFrame, target: str) -> "ModelSelection":
        """Fit models to data and create a list of trained models, `trained_models`.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the data to be used for training.
        target : str
            Name of the target column.

        Returns
        -------
        ModelSelection : object
            The fitted ModelSelection object.

        """
        x = df.copy().drop(target, axis=1)
        y = df.copy()[target]
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(x, y, test_size=0.2)
        for model in self.models:
            model.fit(self.X_train, self.y_train)
        self.metrics = get_metrics(self.models, self.X_test, self.y_test)
        return self

    def explain(self) -> Explanation:
        """Interpret the trained models using Partial Dependence Plots.

        Returns
        -------
        dict : dictionary
            Dictionary of Partial Dependence Plots for each trained model.
        """
        self.explaination = Explanation(self)
        return self.explaination
