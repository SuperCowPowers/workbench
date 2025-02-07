import pandas as pd
from workbench.algorithms.dataframe.proximity import Proximity


class FeatureSpaceProximity(Proximity):
    def __init__(self, model, n_neighbors: int = 10) -> None:
        """
        Initialize the FeatureSpaceProximity class.

        Args:
            model (Model): A Workbench model object.

        """
        from workbench.api import FeatureSet, Endpoint

        # Grab the features and target from the model
        features = model.features()
        target = model.target()

        # Retrieve the training DataFrame from the feature set
        fs = FeatureSet(model.get_input())
        df = fs.view("training").pull_dataframe()

        # Run inference on the model to get the predictions
        end = Endpoint(model.endpoints()[0])
        df = end.inference(df)

        # Call the parent class constructor
        super().__init__(df, id_column=fs.id_column, features=features, target=target, n_neighbors=n_neighbors)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Test a Workbench classification Model
    from workbench.api import Model

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
