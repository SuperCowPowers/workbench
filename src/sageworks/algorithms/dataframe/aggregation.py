"""Aggregation: Perform Row Aggregation on a DataFrame"""

import pandas as pd
import logging

# SageWorks Logger
log = logging.getLogger("sageworks")


def aggregate(df: pd.DataFrame, group_column: str, features: list = None) -> pd.DataFrame:
    """Aggregate Row of a DataFrame
    Args:
        df: Pandas DataFrame
        group_column: The column to aggregate/group on
        features: List of column names (numeric) to perform aggregation on (default: None)
    Returns:
        Pandas DataFrame with aggregated rows and averaged numeric columns
    """

    # If no features are given, indentify all numeric columns
    if features is None:
        features = [x for x in df.select_dtypes(include="number").columns.tolist() if not x.endswith("id")]
        log.info("No features given, auto identifying numeric columns...")
        log.info(f"{features}")

    # Sanity checks
    if not all(column in df.columns for column in features):
        log.critical("Some features are missing in the DataFrame")
        return df
    if df.empty:
        log.critical("DataFrame is empty")
        return df

    # Now aggregate the DataFrame
    log.info(f"Aggregating dataframe, averaging {features}...")

    # Subset the DataFrame to only the features and group column
    df = df[features + [group_column]].copy()
    df["group_count"] = 1

    # Define the aggregation methods for each column
    agg_methods = {feature: "mean" for feature in features}
    agg_methods["group_count"] = "size"

    # Group by the group column and perform the aggregation
    df = df.groupby(group_column).agg(agg_methods).reset_index()

    # Return the DataFrame
    return df


def test():
    """Test for the Row Aggregation Class"""
    # Set some pandas options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Make some fake data
    data = {
        "ID": [
            "id_0",
            "id_0",
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
        "outlier_group": [
            "sample",
            "a_low",
            "sample",
            "b_high",
            "sample",
            "c_high",
            "sample",
            "d_low",
            "sample",
            "e_high",
        ],
    }
    data_df = pd.DataFrame(data)

    # Aggregate the DataFrame
    new_df = aggregate(data_df, group_column="outlier_group")

    # Output the DataFrame
    print(new_df)


if __name__ == "__main__":
    test()
