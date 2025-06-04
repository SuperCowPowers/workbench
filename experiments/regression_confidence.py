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

# Compute SHAP values and get the top 20 features
"""
shap_importances = shap_feature_importance(model)[:20]
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
    "numrotatablebonds",
    "numhacceptors",
    "heavyatomcount",
    "ringcount",
    "numaliphaticrings",
    "numaromaticrings",
    "numsaturatedrings",
]
df = Projection2D().fit_transform(df, features=shap_features, projection="UMAP")


# First the "mixed" cluster
mixed_ids = [
    "A-2232",
    "A-3067",
    "A-690",
    "A-886",
    "B-1540",
    "B-2020",
    "B-2235",
    "B-872",
    "B-873",
    "C-1012",
    "C-1018",
    "C-1037",
    "C-2350",
    "C-2396",
    "C-2449",
    "C-2463",
    "C-948",
    "C-987",
    "F-838",
    "F-999",
]

print(mixed_ids)

low_ids = ["B-3202", "B-4094", "B-3169", "B-3191", "B-4092", "B-4093", "B-2885", "B-3201", "C-718", "H-450", "B-2811"]

# Get a specific set of IDs (neighboring points)
df["neigh"] = df["id"].isin(low_ids).astype(int)

# Run the Unit Test on the Plugin using the new DataFrame with 'x' and 'y'
unit_test = PluginUnitTest(ScatterPlot, input_data=df, x="x", y="y")
unit_test.run()


# Columns that we want to show when we hover above a point
hover_columns = ["q_025", "q_25", "q_50", "q_75", "q_975", "prediction"]

# PluginUnitTest(ScatterPlot, input_data=df, x="x", y="y", color="confidence", hover_columns=hover_columns).run()
unit_test = PluginUnitTest(ScatterPlot, input_data=df, x="x", y="y").run()
unit_test.run()
