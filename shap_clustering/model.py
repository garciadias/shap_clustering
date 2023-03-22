from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMRegressor
from shap import Explainer
from shap.plots import partial_dependence
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


@dataclass
class Explanation:
    """Class for interpreting a selection of trained models."""

    model_selection: "ModelSelection"

    def __post_init__(self):
        self.shap_values_ = self._shap_values()
        self.shap_importance_ = self._shap_importance()
        # self.pdp = self._pdp()

    def _shap_importance(self) -> pd.DataFrame:
        """Return shap importance for the trained models.

        Returns
        -------
        pd.DataFrame : dataframe
            Dataframe of SHAP importance for each trained model.

        """
        shap_importance = []
        for shap_values in self.shap_values_.values():
            shap_importance.append(abs(shap_values).mean(axis=0))
        indexes = self.model_selection.X_train.columns
        cols = list(self.shap_values_.keys())
        return pd.DataFrame(shap_importance, columns=indexes, index=cols).T

    def _pdp(self) -> dict:
        """Interpret the trained models using Partial Dependence Plots.

        Returns
        -------
        dict : dictionary
            Dictionary of Partial Dependence Plots for each trained model.

        """
        pdp = {}
        for feature in self.model_selection.X_train.columns:
            pdp[feature] = {}
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
                pdp[feature][model_name] = plt.gca()
                plt.close()
        return pdp

    def _shap_values(self) -> dict:
        """Return shap values for the trained models.

        Returns
        -------
        dict : dictionary
            Dictionary of SHAP values for each trained model.

        """
        shap_values = {}
        for model in self.model_selection.models:
            model_name = model.__class__.__name__
            X = self.model_selection.X_train
            explainer = Explainer(model, X)
            shap_values[model_name] = explainer.shap_values(X)
        return shap_values


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
        X = df.copy().drop(target, axis=1)
        y = df.copy()[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2
        )
        for model in self.models:
            model.fit(self.X_train, self.y_train)
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
