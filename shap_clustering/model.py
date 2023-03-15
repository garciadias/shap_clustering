from dataclasses import dataclass, field

from pycaret import regression
import matplotlib.pyplot as plt


@dataclass
class ModelSelection:
    n_select: int = 3
    trained_models: field(default_factory=list) = field(init=False)
    setup: field(default_factory=dict) = field(init=False)

    def fit(self, df, target):
        # train model
        self.setup = regression.setup(
            data=df,
            target=target,
            session_id=123,
            silent=True,
            log_experiment=True,
            experiment_name="shap_clustering",
        )
        self.trained_models = regression.compare_models(
            sort="R2", n_select=3, fold=5, verbose=False
        )
        return self

    def interpret_models(self):
        interpretations = {}
        for model in self.trained_models:
            model_name = model.__class__.__name__
            regression.interpret_model(
                model, plot="pdp"
            )
            figure = plt.gcf()
            interpretations[model_name] = figure
        return interpretations
