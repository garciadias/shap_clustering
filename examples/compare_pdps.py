# %%
%cd ..
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from numpy import tile, zeros
from shap import Explanation
from shap.plots import scatter
from sklearn.datasets import load_diabetes

from shap_clustering.model import ModelSelection

# %%
df = load_diabetes(as_frame=True).frame
models = ModelSelection()
models.fit(df, "target")

# %%
explanation = models.explain()
# %%
explanation.importance_plot()
# %%
models.metrics
