from dataclasses import dataclass, field

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from shap import Explainer
from shap.plots import partial_dependence
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils import resample


@dataclass
class Explanation:
    """Class for interpreting a selection of trained models."""

    model_selection: "ModelSelection"

    def __post_init__(self):
        self.pdp = self.pdp()

    #     self.explainers = {}
    #     self.shap_values = {}
    #     for model in self.model_selection.models:
    #         model_name = model.__class__.__name__
    #         explainer = Explainer(model, resample(self.model_selection.X_train))
    #         self.explainers[model_name] = explainer
    #         self.shap_values[model_name] = explainer(
    #             self.model_selection.X_test
    #         )

    def pdp(self) -> dict:
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
                pdp[feature][model_name] = plt.gcf()
                plt.close()
        return pdp


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
            RandomForestRegressor(),
            lgb.LGBMRegressor(),
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
        return Explanation(self)
