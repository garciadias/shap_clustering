# %%
%cd ..
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
print(models.metrics)

# %%
clusters = explanation.cluster_shap_values()
# %%
clusters["LinearRegression"].value_counts()
# %%
# run tsne on the shap values
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

shap_tsne = {}
for model_name in clusters.columns:
    tsne = TSNE(n_components=2, random_state=0)
    feature_columns = df.columns[:-1]
    shap_value_2d = tsne.fit_transform(explanation.shap_values_[model_name][feature_columns])
    shap_tsne[model_name] = shap_value_2d
# %%
fig, axis = plt.subplots(1, 3, figsize=(15, 5))
for i, (model_name, data) in enumerate(shap_tsne.items()):
    axis[i].scatter(data[:, 0], data[:, 1], c=clusters[model_name])
# %%
