"""SQL based Outliers: Compute outliers for all the columns in a DataSource using SQL"""
import logging
import pandas as pd

# SageWorks Imports
from sageworks.artifacts.data_sources.data_source_abstract import DataSourceAbstract
from sageworks.utils.sageworks_logging import logging_setup
from sageworks.utils.pandas_utils import shorten_values

# Setup Logging
logging_setup()
log = logging.getLogger(__name__)


class Outliers:
    """Outliers: Class to compute outliers for all the columns in a DataSource using SQL"""

    def __init__(self):
        """SQLOutliers Initialization"""
        self.outlier_group = "unknown"

    def compute_outliers(
        self,
        data_source: DataSourceAbstract,
        scale: float = 1.25,
        use_stddev: bool = False,
        include_strings: bool = False,
    ) -> pd.DataFrame:
        """Compute outliers for all the numeric columns in a DataSource
        Args:
            data_source(DataSource): The DataSource that we're computing outliers on
            scale(float): The scale to use for either the IQR or stddev outlier calculation (default: 1.25)
            use_stddev(bool): Option to use the standard deviation for the outlier calculation (default: False)
            include_strings(bool): Option to include string columns in the outlier calculation (default: False)
        Returns:
            pd.DataFrame: A DataFrame of outliers for this DataSource
        Notes:
            Uses the IQR * 1.25 (~= 2 Sigma) (use 1.7 for ~= 3 Sigma)
            The scale parameter can be adjusted to change the IQR multiplier
        """
        data_source.log.info("Computing Outliers for numeric columns...")

        # Note: If use_stddev is True, then the scale parameter needs to be adjusted
        if use_stddev and scale == 1.25:  # If the default scale is used, adjust it
            scale = 2.5

        # Compute the numeric outliers
        numeric_outliers_df = self._numeric_outliers(data_source, scale, use_stddev)

        # Compute the string outliers
        if include_strings:
            data_source.log.info("Computing Outliers for string columns...")
            string_outliers_df = self._string_outliers(data_source)
        else:
            string_outliers_df = None

        # Combine the numeric and string outliers
        if numeric_outliers_df is not None and string_outliers_df is not None:
            all_outliers = pd.concat([numeric_outliers_df, string_outliers_df])
        elif numeric_outliers_df is not None:
            all_outliers = numeric_outliers_df
        elif string_outliers_df is not None:
            all_outliers = string_outliers_df
        else:
            log.warning("No outliers found for this DataSource, returning empty DataFrame")
            all_outliers = pd.DataFrame(columns=data_source.column_names() + ["outlier_group"])

        # Drop duplicates
        all_except_outlier_group = [col for col in all_outliers.columns if col != "outlier_group"]
        all_outliers = all_outliers.drop_duplicates(subset=all_except_outlier_group, ignore_index=True)

        # Make sure the dataframe isn't too big, if it's too big sample it down
        if len(all_outliers) > 100:
            log.warning(f"Outliers DataFrame is too large {len(all_outliers)}, sampling down to 100 rows")
            all_outliers = all_outliers.sample(100)

        # Sort by outlier_group and reset the index
        all_outliers = all_outliers.sort_values("outlier_group").reset_index(drop=True)

        # Shorten any long string values
        all_outliers = shorten_values(all_outliers)

        return all_outliers

    def _numeric_outliers(self, data_source: DataSourceAbstract, scale: float, use_stddev=False) -> pd.DataFrame:
        """Internal method to compute outliers for all numeric columns
        Args:
            data_source(DataSource): The DataSource that we're computing outliers on
            scale(float): The scale to use for the IQR outlier calculation
            use_stddev(bool): Option to use the standard deviation for the outlier calculation (default: False)
        Returns:
            pd.DataFrame: A DataFrame of all the outliers combined
        """

        # Grab the column stats and descriptive stats for this DataSource
        column_stats = data_source.column_stats()
        descriptive_stats = data_source.descriptive_stats()

        # For every column in the data_source that is numeric get the outliers
        log.info("Computing outliers for numeric columns (this may take a while)...")
        outlier_df_list = []
        numeric = ["tinyint", "smallint", "int", "bigint", "float", "double", "decimal"]
        for column, data_type in zip(data_source.column_names(), data_source.column_types()):
            if data_type in numeric:
                # Skip columns that are 'binary' columns
                if column_stats[column]["unique"] == 2:
                    log.info(f"Skipping binary column {column}")
                    continue
                if use_stddev:
                    # Compute dataframes for the lower and upper bounds
                    mean = descriptive_stats[column]["mean"]
                    stddev = descriptive_stats[column]["stddev"]
                    lower_bound = mean - (stddev * scale)
                    upper_bound = mean + (stddev * scale)
                    lower_df, upper_df = self._outlier_dfs(data_source, column, lower_bound, upper_bound)
                else:
                    # Compute the IQR for this column
                    iqr = descriptive_stats[column]["q3"] - descriptive_stats[column]["q1"]

                    # Catch cases where IQR is 0
                    if iqr == 0:
                        log.info(f"IQR is 0 for column {column}, but we'll give it a go...")

                    # Compute dataframes for the lower and upper bounds
                    lower_bound = descriptive_stats[column]["q1"] - (iqr * scale)
                    upper_bound = descriptive_stats[column]["q3"] + (iqr * scale)
                    lower_df, upper_df = self._outlier_dfs(data_source, column, lower_bound, upper_bound)

                # If we have outliers, add them to the list
                if lower_df is not None:
                    # Append the outlier DataFrame to the list
                    log.info(f"Found {len(lower_df)} low outliers for column {column}")
                    lower_df["outlier_group"] = f"{column}_low"
                    outlier_df_list.append(lower_df)
                if upper_df is not None:
                    # Append the outlier DataFrame to the list
                    log.info(f"Found {len(upper_df)} high outliers for column {column}")
                    upper_df["outlier_group"] = f"{column}_high"
                    outlier_df_list.append(upper_df)

        # Return the combined DataFrame
        return pd.concat(outlier_df_list) if outlier_df_list else None

    def _string_outliers(self, data_source: DataSourceAbstract) -> pd.DataFrame:
        """Internal method to compute outliers for all the string columns in a DataSource
        Args:
            data_source(DataSource): The DataSource that we're computing outliers on
        Returns:
            pd.DataFrame: A DataFrame of all the outliers combined
        """

        log.info("Computing outliers for string columns (this may take a while)...")
        outlier_df_list = []
        num_rows = data_source.details()["num_rows"]
        outlier_min_count = max(3, num_rows * 0.001)  # 0.1% of the total rows
        value_count_info = data_source.value_counts()
        for column, data_type in zip(data_source.column_names(), data_source.column_types()):
            print(column, data_type)
            # String columns will use the value counts to compute outliers
            if data_type == "string":
                # Skip columns that end with _id or _ip
                if column.endswith("_id") or column.endswith("_ip"):
                    log.info(f"Skipping column {column}")
                    continue
                # Skip columns where all values are unique (all counts are 1)
                if all(value == 1 for value in value_count_info[column].values()):
                    log.info(f"All values are unique for column {column}, skipping...")
                    continue

                # Now loop through the value counts and find any rare values
                for value, count in value_count_info[column].items():
                    if count < outlier_min_count:
                        log.info(f"Found outlier feature {value} for column {column}")
                        query = f"SELECT * from {data_source.table_name} where {column} = '{value}' limit 3"
                        print(query)
                        df = data_source.query(query)
                        df["outlier_group"] = f"{column}_rare"
                        outlier_df_list.append(df)

        # Return the combined DataFrame
        return pd.concat(outlier_df_list) if outlier_df_list else None

    @staticmethod
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
        query = f'SELECT * from {data_source.table_name} where "{column}" < {lower_bound} order by "{column}" limit 10'
        lower_df = data_source.query(query)

        # Check for no results
        if lower_df.shape[0] == 0:
            lower_df = None

        # Get upper outlier bound
        query = (
            f'SELECT * from {data_source.table_name} where "{column}" > {upper_bound} order by "{column}" desc limit 10'
        )
        upper_df = data_source.query(query)

        # Check for no results
        if upper_df.shape[0] == 0:
            upper_df = None

        # Return the lower and upper outlier DataFrames
        return lower_df, upper_df


if __name__ == "__main__":
    """Exercise the SQL Outliers Functionality"""
    from sageworks.artifacts.data_sources.data_source import DataSource

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
