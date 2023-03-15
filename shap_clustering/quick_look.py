# %%
from shap.plots import partial_dependence
from pycaret.datasets import get_data
from pycaret import regression

from shap_clustering.model import ModelSelection

# %%
df = get_data("house")
selection = ModelSelection()
selection.fit(df.sample(100), "SalePrice")
interpretations = selection.interpret_models()

# %%
selection.setup[42]
# %%
