# Workbench Imports
from workbench.api import FeatureSet, Model, Endpoint, DFStore
from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot
from workbench.web_interface.components.plugin_unit_test import PluginUnitTest
from workbench.utils.shap_utils import shap_feature_importance
from workbench.algorithms.dataframe.projection_2d import Projection2D

model = Model("aqsol-ensemble")

# Pull a FeatureSet and run inference on it
recreate = False
if recreate:
    fs = FeatureSet(model.get_input())
    df = fs.pull_dataframe()
    end = Endpoint(model.endpoints()[0])
    df = end.inference(df)

    # Store the inference dataframe
    DFStore().upsert("/workbench/models/aqsol-ensemble/full_inference", df)
else:
    # Retrieve the cached inference dataframe
    df = DFStore().get("/workbench/models/aqsol-ensemble/full_inference")
    if df is None:
        raise ValueError("No cached inference DataFrame found.")

# Compute SHAP values and get the top 10 features
"""
shap_importances = shap_feature_importance(model)[:10]
shap_features = [feature for feature, _ in shap_importances]
"""
shap_features = [
    "mollogp",
    "bertzct",
    "molwt",
    "tpsa",
    "numvalenceelectrons",
    "balabanj",
    "molmr",
    "labuteasa",
    "numhdonors",
    "numheteroatoms",
]
df = Projection2D().fit_transform(df, features=shap_features, projection="UMAP")


# First the "mixed" cluster
mixed_ids = [
    "A-1392",
    "B-162",
    "A-2676",
    "A-2482",
    "A-2152",
    "A-6080",
    "A-238",
    "A-5820",
    "A-2604",
    "A-5686",
    "A-5563",
    "A-5988",
    "A-5851",
    "A-5604",
    "A-6092",
    "A-5589",
    "A-5844",
    "A-2668",
    "A-55",
    "A-3275",
    "A-5086",
]
mixed_big = [
    "A-1392",
    "B-162",
    "A-2482",
    "A-2152",
    "A-6080",
    "A-238",
    "A-5820",
    "A-2604",
    "A-5686",
    "A-5563",
    "A-5988",
    "A-5851",
    "A-5604",
    "A-6092",
    "A-5589",
    "A-5844",
    "A-2668",
    "A-55",
    "A-3275",
    "A-5086",
    "A-3390",
    "A-2234",
    "A-5672",
    "A-343",
    "A-495",
    "A-1974",
    "A-1521",
    "A-5887",
    "A-719",
    "A-2676",
    "A-2765",
]
print(mixed_ids)

# Get a specific set of IDs (neighboring points)
# query_ids = ['C-2383', 'B-976', 'B-866', 'B-867', 'B-868', 'B-3565', 'G-296', 'C-2215', 'B-861', 'B-870', 'B-871']
df["neigh"] = df["id"].isin(mixed_ids).astype(int)

# Run the Unit Test on the Plugin using the new DataFrame with 'x' and 'y'
unit_test = PluginUnitTest(ScatterPlot, input_data=df, x="x", y="y")
unit_test.run()


# Columns that we want to show when we hover above a point
hover_columns = ["q_10", "q_25", "q_50", "q_75", "q_90", "prediction"]

# PluginUnitTest(ScatterPlot, input_data=df, x="x", y="y", color="confidence", hover_columns=hover_columns).run()
unit_test = PluginUnitTest(ScatterPlot, input_data=df, x="x", y="y").run()
unit_test.run()
