import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.cluster import MeanShift

# SageWorks Imports
from sageworks.api import FeatureSet, Model


# Sample code to run DBSCAN
def run_dbscan(df: pd.DataFrame, feature_list: list, eps: float = 0.5, min_samples: int = 5):
    """
    Run DBSCAN on the provided DataFrame using the specified feature list.

    Args:
        df (pd.DataFrame): The input DataFrame.
        feature_list (list): The list of features to use for clustering.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered a core point.

    Returns:
        pd.Series: Cluster labels assigned by DBSCAN.
    """
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_list])

    # Run HDBSCAN
    db = HDBSCAN()
    df["db_cluster"] = db.fit_predict(scaled_features)
    return df


# Sample code to run Mean Shift
def run_mean_shift(df: pd.DataFrame, feature_list: list):
    """
    Run Mean Shift on the provided DataFrame using the specified feature list.

    Args:
        df (pd.DataFrame): The input DataFrame.
        feature_list (list): The list of features to use for clustering.

    Returns:
        pd.Series: Cluster labels assigned by Mean Shift.
    """
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_list])

    # Run Mean Shift
    mean_shift = MeanShift()
    df["mean_shift_cluster"] = mean_shift.fit_predict(scaled_features)
    return df


# Example usage
if __name__ == "__main__":
    # Grab the dataframe from the FeatureSet
    fs = FeatureSet("wine_features")
    df = fs.pull_dataframe()

    # Grab the Feature List from the Model
    model = Model("wine-classification")
    features = model.features()

    # Run DBSCAN and print the cluster labels
    df = run_dbscan(df, features)
    print(df)
    print(df["db_cluster"].value_counts())

    # Run Mean Shift and print the cluster labels
    df = run_mean_shift(df, features)
    print(df)
    print(df["mean_shift_cluster"].value_counts())

    # Grab the dataframe from the FeatureSet
    fs = FeatureSet("abalone_features")
    df = fs.pull_dataframe()

    # Grab the Feature List from the Model
    model = Model("abalone-regression")
    features = model.features()

    # Run DBSCAN and print the cluster labels
    df = run_dbscan(df, features)
    print(df)
    print(df["db_cluster"].value_counts())

    # Run Mean Shift and print the cluster labels
    df = run_mean_shift(df, features)
    print(df)
    print(df["mean_shift_cluster"].value_counts())
