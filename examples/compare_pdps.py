# %%
# %cd ..
from sklearn.datasets import load_diabetes

from shap_clustering.model import ModelSelection

# %%
df = load_diabetes(as_frame=True).frame
rename_dict = {
    "age": "Age",
    "sex": "Sex",
    "bmi": "BMI",
    "bp": "Blood Pressure",
    "s1": "Total Cholesterol",
    "s2": "LDL Cholesterol",
    "s3": "HDL Cholesterol",
    "s4": "Thyroid",
    "s5": "Glaucoma",
    "s6": "Glucose",
}
df = df.rename(columns=rename_dict)
models = ModelSelection()
models.fit(df, "target")

# %%
explanation = models.explain()
# %%
explanation.importance_plot()
# %%
models.metrics
