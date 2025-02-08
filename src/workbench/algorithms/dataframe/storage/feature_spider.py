"""FeatureSpider: A Spider for data/feature investigation and QA"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Feature Spider Class
class FeatureSpider:
    def __init__(
        self,
        df: pd.DataFrame,
        features: list,
        id_column: str,
        target_column: str,
        neighbors: int = 5,
        categorical_target: bool = False,
    ):
        """FeatureSpider: A Spider for data/feature investigation and QA

        Args:
             df: Pandas DataFrame
             features: List of feature column names
             id_column: Name of the ID column
             target_column: Name of the target column
             neighbors: Number of neighbors to use in the KNN model (default: 5)
             categorical_target: Is the target column categorical (default: False)
        """
        # Check for expected columns
        for column in [id_column, target_column] + features:
            if column not in df.columns:
                print(f"DataFrame does not have required {column} Column!")
                return

        # Set internal vars that are used later
        self.df = df.copy()
        self.id_column = id_column
        self.target_column = target_column
        self.features = features
        self.categorical_target = categorical_target

        # Check for NaNs in the features and log the percentage
        for feature in features:
            nan_count = df[feature].isna().sum()
            if nan_count > 0:
                print(f"Feature '{feature}' has {nan_count} NaNs ({nan_count / len(df) * 100:.2f}%).")

        # Remove and NaNs or INFs in the features
        print(f"Dataframe Shape before NaN/INF removal {self.df.shape}")
        self.df[features] = self.df[features].replace([float("inf"), float("-inf")], pd.NA)
        self.df = self.df.dropna(subset=features).reset_index(drop=True)
        print(f"Dataframe Shape after NaN/INF removal {self.df.shape}")

        # Build our KNN model pipeline with StandardScalar
        knn = KNeighborsRegressor(n_neighbors=neighbors, weights="distance")
        self.pipe = make_pipeline(StandardScaler(), knn)

        # Fit Model on features and target
        y = self.df[self.target_column]
        X = self.df[self.features]
        self.pipe.fit(X, y)

        # Grab the Standard Scalar and KNN from the pipeline model
        # Note: These handles need to be constructed after the fit
        self.scalar = self.pipe["standardscaler"]
        self.knn = self.pipe["kneighborsregressor"]

        # This is for collection of the neighbor distances
        self.neigh_distances = []

    def get_feature_matrix(self):
        """Return the KNN Model Internal Feature Matrix"""
        return self.knn._fit_X

    def predict(self, pred_df: pd.DataFrame) -> list:
        """Provide a prediction from the KNN Pipeline model (knn_prediction)"""
        return self.pipe.predict(pred_df[self.features])

    def confidence_scores(self, pred_df: pd.DataFrame, model_preds: pd.Series = None) -> list:
        """Compute Confidence Scores for each Prediction"""

        # Get all the KNN information relevant to this calculation
        neighbor_info = self.neighbor_info(pred_df)

        # Handles for all the relevant info
        knn_preds = neighbor_info["knn_prediction"]
        target_values = neighbor_info["knn_target_values"]
        distances = neighbor_info["knn_distances"]

        # We can score confidence even if we don't have model predictions (less good)
        if model_preds is None:
            model_preds = knn_preds
            stddev_multiplier = 1.5
        else:
            stddev_multiplier = 1.0

        # Now a big loop over all these values to compute the confidence scores
        confidence_scores = []
        for pred, knn_pred, str_val_list, str_dist_list in zip(model_preds, knn_preds, target_values, distances):
            # Each of these is a string of a list (yes a bit cheesy)
            vals = [float(val) for val in str_val_list.split(", ")]
            _ = [float(dis) for dis in str_dist_list.split(", ")]  # dist current not used

            # Compute stddev of the target values
            knn_stddev = np.std(vals)

            # Confidence Score
            conf = 0.5 * (2.0 - abs(float(pred) - float(knn_pred)))
            conf -= knn_stddev * stddev_multiplier

            # Confidence score has min-max of 0-1
            conf = min(max(conf, 0), 1)

            confidence_scores.append(conf)

        # Return the confidence scores
        return confidence_scores

    def neighbor_info(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Provide information on the neighbors (prediction, knn_target_values, knn_distances)"""

        # Make sure we have all the features
        if not set(self.features) <= set(pred_df.columns):
            print(f"DataFrame does not have required features: {self.features}")
            return None

        # Run through scaler
        x_scaled = self.scalar.transform(pred_df[self.features])

        # Add the data to a copy of the dataframe
        results_df = pd.DataFrame()
        results_df["knn_prediction"] = self.knn.predict(x_scaled)

        # Get the Neighbors Information
        neigh_dist, neigh_ind = self.knn.kneighbors(x_scaled)
        target_values = self.knn._y[neigh_ind]

        # Collect the neighbor distances by unpacking the list of lists
        self.neigh_distances = [dist for sublist in neigh_dist for dist in sublist]

        # Note: We're assuming that the Neighbor Index is the same order/cardinality as the dataframe
        results_df["knn_target_values"] = [", ".join([str(val) for val in values]) for values in target_values]
        results_df["knn_distances"] = [", ".join([str(dis) for dis in distances]) for distances in neigh_dist]

        return results_df

    def neighbor_ids(self, pred_df) -> pd.DataFrame:
        """Provide id, features for the neighbors (knn_ids, knn_features)"""

        # Run through scaler
        x_scaled = self.scalar.transform(pred_df[self.features])

        # Add the data to a copy of the dataframe
        results_df = pred_df.copy()

        # Neighbor ID and feature lookups
        neigh_dist, neigh_ind = self.knn.kneighbors(x_scaled)
        results_df["knn_ids"] = [
            ", ".join(self.df.iloc[index][self.id_column] for index in indexes) for indexes in neigh_ind
        ]
        results_df["knn_features"] = [
            ", ".join(self.df.iloc[index][self.id_column] for index in indexes) for indexes in neigh_ind
        ]
        return results_df

    def coincident(self, target_diff: float, verbose: bool = True):
        """Convenience method that returns high_gradients with a distance of 0.0
        Args:
            target_diff(float): The target difference threshold
            verbose(bool): Print out the results (default: True)
        Returns:
            List of indexes that are part of high target gradient (HTG) pairs

        Note: See high_gradients for more information
        """
        return self.high_gradients(0.0, target_diff, verbose)

    def high_gradients(self, within_distance: float, target_diff: float, verbose: bool = True) -> list:
        """Find High Target Gradients in the KNN Model
        Args:
            within_distance(float): The distance threshold to consider
            target_diff(float): The target difference threshold
            verbose(bool): Print out the results (default: True)
        Returns:
            List of indexes that are part of high target gradient (HTG) pairs

        Notes: This basically loops over all the X features in the KNN model
        - Grab the neighbors distances and indices
        - For neighbors `within_distance`* grab target values
        - If target values have a difference > `target_diff`
           - List out the details of the observations and the distance, target diff
        """
        global_htg_set = set()
        for my_index, obs in enumerate(self.knn._fit_X):
            neigh_distances, neigh_indexes = self.knn.kneighbors([obs])
            neigh_distances = neigh_distances[0]  # Returns a list within a list so grab the inner list
            neigh_indexes = neigh_indexes[0]  # Returns a list within a list so grab the inner list
            target_values = self.knn._y[neigh_indexes]

            # Grab the info for this observation
            my_id = self.df.iloc[my_index][self.id_column]
            my_features = self.df.iloc[my_index][self.id_column]
            my_target = self.knn._y[my_index]

            # Loop through the neighbors
            # Note: by definition this observation will be in the neighbors so account for that
            my_htg_set = set()
            for n_index, dist, target in zip(neigh_indexes, neigh_distances, target_values):
                # Skip myself
                if n_index == my_index:
                    continue

                # Do we have a categorical target (not numeric)?
                if self.categorical_target:
                    _diff = 1.0 if my_target != target else 0.0
                else:
                    _diff = abs(my_target - target)

                # Compute target differences `within_distance` feature space
                if dist <= within_distance and _diff >= target_diff:
                    # Update the individual HTG set for this observation
                    my_htg_set.add((n_index, dist, _diff, target))

                    # Add both (me and my neighbor) to the global high gradient index list
                    global_htg_set.add(my_index)
                    global_htg_set.add(n_index)

            # Okay we've computed our HTG set for this observation
            # Print out all my HTG neighbors if the verbose flag is set
            if verbose and my_htg_set:
                print(f"\nOBSERVATION: {my_id}")
                print(f"\t{my_id}({my_target:.2f}):{my_features}")
                for htg_neighbor, dist, _diff, target in my_htg_set:
                    neighbor_id = self.df.iloc[htg_neighbor][self.id_column]
                    neighbor_features = self.df.iloc[htg_neighbor][self.id_column]
                    print(f"\t{neighbor_id}({target:.2f}):{neighbor_features} TargetD:{_diff:.2f} FeatureD:{dist}")

        # Return the global list of indexes that are part of high target gradient (HTG) pairs
        return list(global_htg_set)


def test():
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
        "feat3": [0.1, 0.1, 0.2, 1.6, 2.5, 0.1, 0.1, 0.2, 1.6, 2.5],
        "price": [31, 60, 62, 40, 20, 31, 61, 60, 40, 20],
    }
    data_df = pd.DataFrame(data)

    # Create the class and run the taggers
    f_spider = FeatureSpider(data_df, ["feat1", "feat2", "feat3"], id_column="ID", target_column="price")
    knn_preds = f_spider.predict(data_df)
    print(knn_preds)
    coincident = f_spider.coincident(2)
    print("COINCIDENT")
    print(coincident)
    high_gradients = f_spider.high_gradients(2, 2)
    print("\nHIGH GRADIENTS")
    print(high_gradients)

    # Run some neighbor methods
    query_df = data_df[data_df["ID"] == "id_0"].copy()
    print(f_spider.neighbor_info(query_df))

    # Feature matrix
    print(f_spider.get_feature_matrix())

    # Fake predictions
    predictions = [31, 60, 62, 40, 20, 31, 61, 60, 40, 20]

    # Now get confidence scores
    data_df["confidence"] = f_spider.confidence_scores(data_df, model_preds=predictions)

    # Show a scatter plot of the confidence scores
    from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Columns of Interest
    dropdown_columns = ["feat1", "feat2", "feat3", "price", "confidence"]

    # Run the Unit Test on the Plugin
    unit_test = PluginUnitTest(
        ScatterPlot,
        input_data=data_df[dropdown_columns][:100],
        x="feat1",
        y="feat2",
        color="confidence",
        dropdown_columns=dropdown_columns,
    )
    unit_test.run()


def integration_test():
    """Integration Test for the FeatureResolution Class"""
    from workbench.api.feature_set import FeatureSet
    from workbench.api.model import Model, Endpoint

    # Grab a test dataframe
    fs = FeatureSet("aqsol_features")
    feature_df = fs.pull_dataframe()

    # Get the target and feature columns
    m = Model("aqsol-regression")
    target_column = m.target()
    feature_columns = m.features()

    # Create the class and run the report
    feature_spider = FeatureSpider(
        feature_df, features=feature_columns, target_column=target_column, id_column=fs.id_column, neighbors=2
    )
    feature_spider.coincident(1.0)

    # Now run predictions on the endpoint
    endpoint = Endpoint("aqsol-regression-end")
    pred_df = endpoint.inference(feature_df)
    print(len(pred_df))

    # Now get confidence scores
    pred_df["confidence"] = feature_spider.confidence_scores(feature_df, model_preds=pred_df["prediction"])

    # Show a scatter plot of the confidence scores
    from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Columns of Interest
    dropdown_columns = ["residuals_abs", "prediction", "solubility", "confidence"]

    # Run the Unit Test on the Plugin
    unit_test = PluginUnitTest(
        ScatterPlot,
        input_data=pred_df[dropdown_columns][:100],
        x="solubility",
        y="prediction",
        color="confidence",
        dropdown_columns=dropdown_columns,
    )
    unit_test.run()


if __name__ == "__main__":
    test()
    # integration_test()
