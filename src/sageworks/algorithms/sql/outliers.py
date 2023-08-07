"""SQL based Outliers: Compute outliers for all the numeric columns in a DataSource using SQL"""
import logging
import pandas as pd

# SageWorks Imports
from sageworks.artifacts.data_sources.data_source_abstract import DataSourceAbstract
from sageworks.utils.sageworks_logging import logging_setup

# Setup Logging
logging_setup()
log = logging.getLogger(__name__)


def _outlier_dfs(data_source: DataSourceAbstract, column: str, lower_bound: float, upper_bound: float):
    """Internal method to compute outliers for a numeric column
    Args:
        data_source(DataSource): The DataSource that we're computing outliers on
        column(str): The column to compute outliers on
        lower_bound(float): The lower bound for outliers
        upper_bound(float): The upper bound for outliers
    Returns:
        (pd.DataFrame, pd.DataFrame): A DataFrame for lower outliers and a DataFrame for upper outliers
    """

    # Get lower outlier bound
    query = f"SELECT * from {data_source.table_name} where {column} < {lower_bound} order by {column} limit 5"
    lower_df = data_source.query(query)

    # Check for no results
    if lower_df.shape[0] == 0:
        lower_df = None

    # Get upper outlier bound
    query = f"SELECT * from {data_source.table_name} where {column} > {upper_bound} order by {column} desc limit 5"
    upper_df = data_source.query(query)

    # Check for no results
    if upper_df.shape[0] == 0:
        upper_df = None

    # Return the lower and upper outlier DataFrames
    return lower_df, upper_df


def outliers(data_source: DataSourceAbstract, scale: float = 1.7) -> pd.DataFrame:
    """Compute outliers for all the numeric columns in a DataSource
    Args:
        data_source(DataSource): The DataSource that we're computing outliers on
        scale(float): The scale to use for the IQR (default: 1.7)
    Returns:
        pd.DataFrame: A DataFrame of outliers for this DataSource
    Notes:
        Uses the IQR * 1.7 (~= 3 Sigma) method to compute outliers
        The scale parameter can be adjusted to change the IQR multiplier
    """

    # Grab the quartiles for this DataSource
    quartiles = data_source.quartiles()

    # For every column in the table that is numeric get the outliers
    log.info("Computing outliers for all columns (this may take a while)...")
    outlier_group = 0
    outlier_df_list = []
    outlier_features = []
    num_rows = data_source.details()["num_rows"]
    outlier_min_count = max(3, num_rows * 0.005)  # 0.5% of the total rows
    max_unique_values = 40  # 40 is the max number of value counts that are stored in AWS
    value_count_info = data_source.value_counts()
    for column, data_type in zip(data_source.column_names(), data_source.column_types()):
        print(column, data_type)
        # String columns will use the value counts to compute outliers
        if data_type == "string":
            # Skip columns with too many unique values
            if len(value_count_info[column]) >= max_unique_values:
                log.warning(f"Skipping column {column} too many unique values")
                continue
            for value, count in value_count_info[column].items():
                if count < outlier_min_count:
                    log.info(f"Found outlier feature {value} for column {column}")
                    query = f"SELECT * from {data_source.table_name} where {column} = '{value}' limit 3"
                    print(query)
                    df = data_source.query(query)
                    df["outlier_group"] = outlier_group
                    outlier_group += 1
                    outlier_df_list.append(df)

        elif data_type in ["bigint", "double", "int", "smallint", "tinyint"]:
            iqr = quartiles[column]["q3"] - quartiles[column]["q1"]

            # Catch cases where IQR is 0
            if iqr == 0:
                log.info(f"IQR is 0 for column {column}, skipping...")
                continue

            # Compute dataframes for the lower and upper bounds
            lower_bound = quartiles[column]["q1"] - (iqr * scale)
            upper_bound = quartiles[column]["q3"] + (iqr * scale)
            lower_df, upper_df = _outlier_dfs(data_source, column, lower_bound, upper_bound)

            # If we have outliers, add them to the list
            for df in [lower_df, upper_df]:
                if df is not None:
                    # Add the outlier_group identifier
                    df["outlier_group"] = outlier_group
                    outlier_group += 1

                    # Append the outlier DataFrame to the list
                    log.info(f"Found {len(df)} outliers for column {column}")
                    outlier_df_list.append(df)
                    outlier_features.append(column)

    # Combine all the outlier DataFrames
    if outlier_df_list:
        outlier_df = pd.concat(outlier_df_list)
    else:
        log.warning("No outliers found for this DataSource, returning empty DataFrame")
        outlier_df = pd.DataFrame(columns=data_source.column_names() + ["outlier_group"])

    # Return the outlier dataframe
    return outlier_df


if __name__ == "__main__":
    """Exercise the SQL Outliers Functionality"""
    from sageworks.artifacts.data_sources.data_source import DataSource

    # Setup Pandas output options
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)

    # Retrieve a Data Source
    my_data = DataSource("abalone_data")

    # Verify that the Athena Data Source exists
    assert my_data.check()

    # What's my SageWorks UUID
    print(f"UUID: {my_data.uuid}")

    # Get outliers for numeric columns
    my_outlier_df = outliers(my_data)
    print("\nOutliers")
    print(my_outlier_df)
