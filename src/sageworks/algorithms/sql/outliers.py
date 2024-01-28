"""SQL based Outliers: Compute outliers for all the columns in a DataSource using SQL"""

import logging
import pandas as pd

# SageWorks Imports
from sageworks.core.artifacts.data_source_abstract import DataSourceAbstract

from sageworks.utils.pandas_utils import shorten_values

# SageWorks Logger
log = logging.getLogger("sageworks")


class Outliers:
    """Outliers: Class to compute outliers for all the columns in a DataSource using SQL"""

    def __init__(self):
        """SQLOutliers Initialization"""
        self.outlier_group = "unknown"

    def compute_outliers(
        self, data_source: DataSourceAbstract, scale: float = 1.5, use_stddev: bool = False
    ) -> pd.DataFrame:
        """Compute outliers for all the numeric columns in a DataSource
        Args:
            data_source(DataSource): The DataSource that we're computing outliers on
            scale (float): The scale to use for either the IQR or stddev outlier calculation (default: 1.5)
            use_stddev (bool): Option to use the standard deviation for the outlier calculation (default: False)
        Returns:
            pd.DataFrame: A DataFrame of outliers for this DataSource
        Notes:
            Uses the IQR * 1.5 (~= 2.5 Sigma) (use 1.7 for ~= 3 Sigma)
            The scale parameter can be adjusted to change the IQR multiplier
        """

        # Note: If use_stddev is True, then the scale parameter needs to be adjusted
        if use_stddev and scale == 1.5:  # If the default scale is used, adjust it
            scale = 2.5

        # Compute the numeric outliers
        outlier_df = self._numeric_outliers(data_source, scale, use_stddev)

        # If there are no outliers, return a DataFrame with defined columns but no rows
        if outlier_df is None:
            return pd.DataFrame(columns=data_source.column_names() + ["outlier_group"])

        # Get the top N outliers for each outlier group
        outlier_df = self.get_top_n_outliers(outlier_df)

        # Drop duplicates
        all_except_outlier_group = [col for col in outlier_df.columns if col != "outlier_group"]
        outlier_df = outlier_df.drop_duplicates(subset=all_except_outlier_group, ignore_index=True)

        # Make sure the dataframe isn't too big, if it's too big sample it down
        if len(outlier_df) > 300:
            log.important(f"Outliers DataFrame is too large {len(outlier_df)}, sampling down to 300 rows")
            outlier_df = outlier_df.sample(300)

        # Sort by outlier_group and reset the index
        outlier_df = outlier_df.sort_values("outlier_group").reset_index(drop=True)

        # Shorten any long string values
        outlier_df = shorten_values(outlier_df)
        return outlier_df

    def _numeric_outliers(self, data_source: DataSourceAbstract, scale: float, use_stddev=False) -> pd.DataFrame:
        """Internal method to compute outliers for all numeric columns
        Args:
            data_source(DataSource): The DataSource that we're computing outliers on
            scale (float): The scale to use for the IQR outlier calculation
            use_stddev (bool): Option to use the standard deviation for the outlier calculation (default: False)
        Returns:
            pd.DataFrame: A DataFrame of all the outliers combined
        """

        # Grab the column stats and descriptive stats for this DataSource
        column_stats = data_source.column_stats()
        descriptive_stats = data_source.descriptive_stats()

        # Get the column names and types from the DataSource
        column_details = data_source.column_details(view="computation")

        # For every column in the data_source that is numeric get the outliers
        # This loop computes the columns, lower bounds, and upper bounds for the SQL query
        log.info("Computing Outliers for numeric columns...")
        numeric = ["tinyint", "smallint", "int", "bigint", "float", "double", "decimal"]
        columns = []
        lower_bounds = []
        upper_bounds = []
        for column, data_type in column_details.items():
            if data_type in numeric:
                # Skip columns that just have one value (or are all nans)
                if column_stats[column]["unique"] <= 1:
                    log.info(f"Skipping unary column {column} with value {descriptive_stats[column]['min']}")
                    continue

                # Skip columns that are 'binary' columns
                if column_stats[column]["unique"] == 2:
                    log.info(f"Skipping binary column {column}")
                    continue

                # Do they want to use the stddev instead of IQR?
                if use_stddev:
                    mean = descriptive_stats[column]["mean"]
                    stddev = descriptive_stats[column]["stddev"]
                    lower_bound = mean - (stddev * scale)
                    upper_bound = mean + (stddev * scale)

                # Compute the IQR for this column
                else:
                    iqr = descriptive_stats[column]["q3"] - descriptive_stats[column]["q1"]
                    lower_bound = descriptive_stats[column]["q1"] - (iqr * scale)
                    upper_bound = descriptive_stats[column]["q3"] + (iqr * scale)

                # Add the column, lower bound, and upper bound to the lists
                columns.append(column)
                lower_bounds.append(lower_bound)
                upper_bounds.append(upper_bound)

        # Compute the SQL query
        query = self._multi_column_outlier_query(data_source, columns, lower_bounds, upper_bounds)
        outlier_df = data_source.query(query)

        # Label the outlier groups
        outlier_df = self._label_outlier_groups(outlier_df, columns, lower_bounds, upper_bounds)
        return outlier_df

    @staticmethod
    def _multi_column_outlier_query(
        data_source: DataSourceAbstract, columns: list, lower_bounds: list, upper_bounds: list
    ) -> str:
        """Internal method to compute outliers for multiple columns
        Args:
            data_source(DataSource): The DataSource that we're computing outliers on
            columns(list): The columns to compute outliers on
            lower_bounds(list): The lower bounds for outliers
            upper_bounds(list): The upper bounds for outliers
        Returns:
            str: A SQL query to compute outliers for multiple columns
        """
        # Grab the  DataSource table name
        table = data_source.get_table_name()

        # Get the column names and types from the DataSource
        column_details = data_source.column_details(view="computation")
        sql_columns = ", ".join([f'"{col}"' for col in column_details.keys()])

        query = f"SELECT {sql_columns} FROM {table} WHERE "
        for col, lb, ub in zip(columns, lower_bounds, upper_bounds):
            query += f"({col} < {lb} OR {col} > {ub}) OR "
        query = query[:-4]

        # Add a limit just in case
        query += " LIMIT 5000"
        return query

    @staticmethod
    def _label_outlier_groups(
        outlier_df: pd.DataFrame, columns: list, lower_bounds: list, upper_bounds: list
    ) -> pd.DataFrame:
        """Internal method to label outliers by group.
        Args:
            outlier_df(pd.DataFrame): The DataFrame of outliers
            columns(list): The columns for which to compute outliers
            lower_bounds(list): The lower bounds for each column
            upper_bounds(list): The upper bounds for each column
        Returns:
            pd.DataFrame: A DataFrame with an added 'outlier_group' column, indicating the type of outlier.
        """

        column_outlier_dfs = []
        for col, lb, ub in zip(columns, lower_bounds, upper_bounds):
            mask_low = outlier_df[col] < lb
            mask_high = outlier_df[col] > ub

            low_df = outlier_df[mask_low].copy()
            low_df["outlier_group"] = f"{col}_low"

            high_df = outlier_df[mask_high].copy()
            high_df["outlier_group"] = f"{col}_high"

            column_outlier_dfs.extend([low_df, high_df])

        return pd.concat(column_outlier_dfs, ignore_index=True)

    @staticmethod
    def get_top_n_outliers(outlier_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Function to retrieve the top N highest and lowest outliers for each outlier group.

        Args:
            outlier_df (pd.DataFrame): The DataFrame of outliers with 'outlier_group' column
            n (int): Number of top outliers to retrieve for each group, defaults to 10

        Returns:
            pd.DataFrame: A DataFrame containing the top N outliers for each outlier group
        """

        def get_extreme_values(group):
            """Helper function to get the top N extreme values from a group."""

            # Get the column and extreme type (high or low)
            col, extreme_type = group.name.rsplit("_", 1)

            # Sort values depending on whether they are 'high' or 'low' outliers
            group = group.sort_values(by=col, ascending=(extreme_type == "low"))

            return group.head(n)

        # Group by 'outlier_group' and apply the helper function to get top N extreme values
        top_outliers = outlier_df.groupby("outlier_group").apply(get_extreme_values).reset_index(drop=True)
        return top_outliers


if __name__ == "__main__":
    """Exercise the SQL Outliers Functionality"""
    from sageworks.api.data_source import DataSource

    # Setup Pandas output options
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)

    # Retrieve a Data Source
    my_data = DataSource("test_data")

    # Verify that the Athena Data Source exists
    assert my_data.exists()

    # Create the class and Compute outliers
    my_outliers = Outliers()
    my_outlier_df = my_outliers.compute_outliers(my_data)
    print("\nOutliers")
    print(my_outlier_df)

    # Uncomment this to use the stddev instead of IQR
    # my_outlier_df = my_outliers.compute_outliers(my_data, use_stddev=True)
    # print("\nOutliers (using stddev)")
    # print(my_outlier_df)
