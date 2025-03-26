import pandas as pd
import logging

# Workbench Imports
from workbench.algorithms.dataframe.proximity import Proximity
from workbench.algorithms.dataframe.projection_2d import Projection2D
from workbench.core.views.inference_view import InferenceView
from workbench.api import FeatureSet, Model

# Set up logging
log = logging.getLogger("workbench")


class FeatureSpaceProximity(Proximity):
    def __init__(self, model: Model, n_neighbors: int = 10) -> None:
        """
        Initialize the FeatureSpaceProximity class.

        Args:
            model (Model): A Workbench model object.
            n_neighbors (int): Number of neighbors to compute. Defaults to 10.
        """

        # Grab the features and target from the model
        features = model.features()
        target = model.target()

        # Grab the feature set for the model
        fs = FeatureSet(model.get_input())

        # If we have a "inference" view, pull the data from that view
        view_name = f"inf_{model.uuid.replace('-', '_')}"
        if view_name in fs.views():
            self.df = fs.view(view_name).pull_dataframe()

        # Otherwise, pull the data from the feature set and run inference
        else:
            inf_view = InferenceView.create(model)
            self.df = inf_view.pull_dataframe()

        # Call the parent class constructor
        super().__init__(self.df, id_column=fs.id_column, features=features, target=target, n_neighbors=n_neighbors)

        # Project the data to 2D
        self.df = Projection2D().fit_transform(self.df, features=features)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Test a Workbench classification Model
    m = Model("wine-classification")
    fsp = FeatureSpaceProximity(m)

    # Neighbors Test using a single row from FeatureSet
    fs = FeatureSet(m.get_input())
    df = fs.pull_dataframe()
    single_query_neighbors = fsp.neighbors(df.iloc[[0]])
    print("\nNeighbors for Query ID:", df.iloc[0][fs.id_column])
    print(single_query_neighbors)

    # Test a Workbench regression model
    m = Model("abalone-regression")
    fsp = FeatureSpaceProximity(m)

    # Neighbors Test using a multiple rows from FeatureSet
    fs = FeatureSet(m.get_input())
    df = fs.pull_dataframe()
    query_neighbors = fsp.neighbors(df.iloc[0:2])
    print("\nNeighbors for Query ID:", df.iloc[0][fs.id_column])
    print(query_neighbors)

    # Test a Workbench regression model
    m = Model("aqsol-regression")
    fsp = FeatureSpaceProximity(m)

    # Neighbors Test using a multiple rows from FeatureSet
    fs = FeatureSet(m.get_input())
    df = fs.pull_dataframe()
    query_neighbors = fsp.neighbors(df.iloc[5:7])
    print("\nNeighbors for Query ID:", df.iloc[5][fs.id_column])
    print(query_neighbors)

    # Time the all_neighbors method
    import time

    start_time = time.time()
    all_neighbors_df = fsp.all_neighbors()
    end_time = time.time()
    print("\nTime taken for all_neighbors:", end_time - start_time)
    print("\nAll Neighbors DataFrame:")
    print(all_neighbors_df)

    # Show a scatter plot of the data
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest
    from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot

    # Run the Unit Test on the Plugin using the new DataFrame with 'x' and 'y'
    unit_test = PluginUnitTest(ScatterPlot, input_data=fsp.df, x="x", y="y")
    unit_test.run()
