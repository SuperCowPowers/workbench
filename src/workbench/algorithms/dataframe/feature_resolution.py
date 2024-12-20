"""FeatureResolution: Report on Feature Space Resolution Issues"""

import logging
from typing import Union
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Workbench Imports
from workbench.utils.pandas_utils import DataFrameBuilder


# Feature Resolution Class
class FeatureResolution:
    def __init__(
        self,
        input_df: pd.DataFrame,
        features: list,
        target_column: str,
        id_column: str,
        distance_metric: str = "minkowski",
    ):
        """Initialize the FeatureResolution object

        Args:
            distance_metric: Distance metric to use (default: "minkowski")
        """
        self.log = logging.getLogger("workbench")
        self.df = input_df.copy().reset_index(drop=True)
        self.features = features
        self.target_column = target_column
        self.id_column = id_column
        self.n_neighbors = 10
        self.scalar = StandardScaler()
        self.knn = KNeighborsRegressor(metric=distance_metric, n_neighbors=self.n_neighbors, weights="distance")
        self.dataframe_builder = DataFrameBuilder()
        self.recursive_df_list = []

    def compute(
        self, within_distance: float, min_target_difference: float, output_columns: list = [], verbose=True
    ) -> Union[pd.DataFrame, None]:
        """FeatureResolution: Compute Feature Space to Target Resolution and Report Issues

        Args:
            within_distance: Features within this distance should have similar target values
            min_target_difference: Minimum target difference to consider
            output_columns: List of additional columns to output (default: []])
            verbose: Whether to print out the resolution issues (default: True)

        Returns:
            Pandas DataFrame of Feature Space to Target Resolution Issues
            Includes any additional output_columns if specified
        """

        # Check for expected columns
        for column in [self.target_column] + self.features:
            if column not in self.df.columns:
                self.log.error(f"DataFrame does not have required {column} Column!")
                return

        # Set up the output columns (add id and target columns if they are not already included)
        output_columns = list(set(output_columns).union({self.id_column, self.target_column}))

        # Check the output columns
        if output_columns is not None:
            for column in output_columns:
                if column not in self.df.columns:
                    self.log.error(f"DataFrame does not have required {column} Column!")
                    return

        # Check for NaNs in the features and log the percentage
        for feature in self.features:
            nan_count = self.df[feature].isna().sum()
            if nan_count > 0:
                print(f"Feature '{feature}' has {nan_count} NaNs ({nan_count / len(self.df) * 100:.2f}%).")

        # Remove and NaNs or INFs in the features
        self.log.info(f"Dataframe Shape before NaN/INF removal {self.df.shape}")
        self.df = self.df.replace([float("inf"), float("-inf")], pd.NA).dropna().reset_index(drop=True)
        self.log.info(f"Dataframe Shape after NaN/INF removal {self.df.shape}")

        # Standardize the features
        X = self.scalar.fit_transform(self.df[self.features])
        y = self.df[self.target_column]

        # Fit the KNN model
        self.knn.fit(X, y)

        # Compute the feature space to target resolution to the nearest neighbors
        output_count = 0
        for my_index, row in enumerate(X):
            # Find the nearest neighbors
            distances, indices = self.knn.kneighbors([row])
            distances = distances[0]  # Returns a list within a list so grab the inner list
            indices = indices[0]
            target_values = y[indices]

            # Grab the info for this observation
            my_id = self.df.iloc[my_index][self.id_column]
            my_output_data = self.df.iloc[my_index][output_columns]
            my_target = y[my_index]

            # Loop through the neighbors
            for n_index, n_distance, n_target in zip(indices, distances, target_values):
                # Skip myself
                if n_index == my_index:
                    continue

                # Compute the difference in feature space and target space
                feature_diff = n_distance
                target_diff = abs(my_target - n_target)

                # Compute target differences `within_distance` feature space
                if feature_diff <= within_distance and target_diff >= min_target_difference:
                    # Gather info about the neighbor
                    neighbor_id = self.df.iloc[n_index][self.id_column]
                    neighbor_output_data = self.df.iloc[n_index][output_columns]

                    # Add to the output DataFrame
                    row_data = my_output_data.to_dict()
                    row_data["feature_diff"] = feature_diff
                    row_data["target_diff"] = target_diff
                    row_data["n_id"] = neighbor_id
                    self.dataframe_builder.add_row(row_data)

                    # Print out the resolution issue (if verbose)
                    if verbose:
                        print(f"{output_count} Feature Diff: {feature_diff} Target Diff: {target_diff}")
                        print(f"\t{my_id}: {my_target:.3f} {list(my_output_data)}")
                        print(f"\t{neighbor_id}: {n_target:.3f} {list(neighbor_output_data)}")
                    # Increment the output count
                    output_count += 1

        # Return the output DataFrame
        return self.dataframe_builder.build()

    def recursive_compute(
        self, within_distance: float, min_target_difference: float, output_columns: list = [], verbose=True
    ) -> pd.DataFrame:
        """Compute Feature Resolution Issues, remove the issues, and recurse until no issues are found"""

        # Compute the resolution issues
        resolution_df = self.compute(within_distance, min_target_difference, output_columns, verbose)
        self.recursive_df_list.append(resolution_df)

        # If there are no resolution issues, return the combined DataFrame
        if len(resolution_df) == 0:
            return pd.concat(self.recursive_df_list)

        # Gather all IDs to be removed
        ids_to_remove = set(list(resolution_df[self.id_column]) + list(resolution_df["n_id"]))

        # Remove the rows of the observations that had issues
        print("Removing IDs: ", ids_to_remove)
        self.df = self.df[~self.df[self.id_column].isin(ids_to_remove)]

        # Recurse
        print("Recursing...")
        self.df = self.df.reset_index(drop=True)
        self.dataframe_builder = DataFrameBuilder()
        return self.recursive_compute(within_distance, min_target_difference, output_columns, verbose)


# Test the FeatureResolution Class
def simple_unit_test():
    """Test for the Feature Spider Class"""
    # Set some pandas options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Make some fake data
    data = {
        "ID": [
            "id_0",
            "id_1",
            "id_2",
            "id_3",
            "id_4",
            "id_5",
            "id_6",
            "id_7",
            "id_8",
            "id_9",
        ],
        "feat1": [1.0, 1.0, 1.1, 3.0, 4.0, 1.0, 1.0, 1.1, 3.0, 4.0],
        "feat2": [1.0, 1.0, 1.1, 3.0, 4.0, 1.0, 1.0, 1.1, 3.0, 4.0],
        "feat3": [0.1, 0.2, 0.2, 1.6, 2.5, 0.1, 0.3, 0.2, 1.6, 2.5],
        "price": [10, 11, 12, 40, 20, 35, 61, 60, 40, 20],
    }
    data_df = pd.DataFrame(data)

    # Create the class and run the report
    resolution = FeatureResolution(data_df, features=["feat1", "feat2", "feat3"], target_column="price", id_column="ID")
    resolution.compute(within_distance=0.1, min_target_difference=10)


def unit_test():
    """Unit Test for the FeatureResolution Class"""
    from workbench.api.feature_set import FeatureSet
    from workbench.api.model import Model

    # Grab a test dataframe
    fs = FeatureSet("aqsol_mol_descriptors")
    test_df = fs.pull_dataframe()

    # Get the target and feature columns
    m = Model("aqsol-mol-regression")
    target_column = m.target()
    feature_columns = m.features()

    # Create the class and run the report
    resolution = FeatureResolution(
        test_df, features=feature_columns, target_column=target_column, id_column=fs.id_column
    )
    df = resolution.compute(within_distance=0.01, min_target_difference=1.0)
    print(df)


def recursive_test():
    from workbench.api.feature_set import FeatureSet
    from workbench.api.model import Model

    # Grab a test dataframe
    fs = FeatureSet("aqsol_mol_descriptors")
    test_df = fs.pull_dataframe()

    # Get the target and feature columns
    m = Model("aqsol-mol-regression")
    target_column = m.target()
    feature_columns = m.features()

    # Create the class and run the report
    resolution = FeatureResolution(
        test_df, features=feature_columns, target_column=target_column, id_column=fs.id_column
    )
    df = resolution.recursive_compute(within_distance=0.01, min_target_difference=1.0)
    print(df)


if __name__ == "__main__":
    simple_unit_test()
    unit_test()
    # recursive_test()
