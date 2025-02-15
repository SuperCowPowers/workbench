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
    proximity = FeatureSpaceProximity(m)

    # Neighbors Test using a single query ID
    single_query_id = 5
    single_query_neighbors = proximity.neighbors(single_query_id)
    print("\nNeighbors for Query ID:", single_query_id)
    print(single_query_neighbors)

    # Test a Workbench regression model
    m = Model("abalone-regression")
    proximity = FeatureSpaceProximity(m)

    # Neighbors Test using a single query ID
    single_query_id = 5
    single_query_neighbors = proximity.neighbors(single_query_id)
    print("\nNeighbors for Query ID:", single_query_id)
    print(single_query_neighbors)

    # Show a scatter plot of the data
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest
    from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot

    # Run the Unit Test on the Plugin using the new DataFrame with 'x' and 'y'
    unit_test = PluginUnitTest(ScatterPlot, input_data=proximity.df, x="x", y="y")
    unit_test.run()
