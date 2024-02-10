"""DataSourceAbstract: Abstract Base Class for all data sources (S3: CSV, JSONL, Parquet, RDS, etc)"""

from abc import abstractmethod
import pandas as pd
from io import StringIO
import time

# SageWorks Imports
from sageworks.core.artifacts.artifact import Artifact


class DataSourceAbstract(Artifact):
    def __init__(self, data_uuid: str, database: str = "sageworks"):
        """DataSourceAbstract: Abstract Base Class for all data sources
        Args:
            data_uuid(str): The UUID for this Data Source
            database(str): The database to use for this Data Source (default: sageworks)
        """

        # Call superclass init
        super().__init__(data_uuid)

        # Set up our instance attributes
        self._database = database
        self._table_name = data_uuid
        self._display_columns = None

    def __post_init__(self):
        # Call superclass post_init
        super().__post_init__()

    def get_database(self) -> str:
        """Get the database for this Data Source"""
        return self._database

    def get_table_name(self) -> str:
        """Get the base table name for this Data Source"""
        return self._table_name

    @abstractmethod
    def num_rows(self) -> int:
        """Return the number of rows for this Data Source"""
        pass

    @abstractmethod
    def num_columns(self) -> int:
        """Return the number of columns for this Data Source"""
        pass

    @abstractmethod
    def column_names(self) -> list[str]:
        """Return the column names for this Data Source"""
        pass

    @abstractmethod
    def column_types(self) -> list[str]:
        """Return the column types for this Data Source"""
        pass

    def column_details(self, view: str = "all") -> dict:
        """Return the column details for this Data Source
        Args:
            view (str): The view to get column details for (default: "all")
        Returns:
            dict: The column details for this Data Source
        """
        names = self.column_names()
        types = self.column_types()
        if view == "display":
            return {name: type_ for name, type_ in zip(names, types) if name in self.get_display_columns()}
        elif view == "computation":
            return {name: type_ for name, type_ in zip(names, types) if name in self.get_computation_columns()}
        elif view == "all":
            return {name: type_ for name, type_ in zip(names, types)}  # Return the full column details
        else:
            raise ValueError(f"Unknown column details view: {view}")

    def get_display_columns(self) -> list[str]:
        """Get the display columns for this Data Source
        Returns:
            list[str]: The display columns for this Data Source
        """
        # Check if we have the display columns in our metadata
        if self._display_columns is None:
            self._display_columns = self.sageworks_meta().get("sageworks_display_columns")

        # If we still don't have display columns, try to set them
        if self._display_columns is None:
            # Exclude these automatically generated columns
            exclude_columns = ["write_time", "api_invocation_time", "is_deleted", "event_time", "id"]

            # We're going to remove any excluded columns from the display columns and limit to 30 total columns
            self._display_columns = [col for col in self.column_names() if col not in exclude_columns][:30]

            # Add the outlier_group column if it exists and isn't already in the display columns
            if "outlier_group" in self.column_names():
                self._display_columns = list(set(self._display_columns) + set(["outlier_group"]))

            # Set the display columns in the metadata
            self.set_display_columns(self._display_columns, onboard=False)

        # Return the display columns
        return self._display_columns

    def set_display_columns(self, display_columns: list[str], onboard: bool = True):
        """Set the display columns for this Data Source

        Args:
            display_columns (list[str]): The display columns for this Data Source
            onboard (bool): Onboard the Data Source after setting the display columns (default: True)
        """
        self.log.important(f"Setting Display Columns...{display_columns}")
        self._display_columns = display_columns
        self.upsert_sageworks_meta({"sageworks_display_columns": self._display_columns})
        if onboard:
            self.onboard()

    def num_display_columns(self) -> int:
        """Return the number of display columns for this Data Source"""
        return len(self._display_columns) if self._display_columns else 0

    def get_computation_columns(self) -> list[str]:
        return self.get_display_columns()

    def set_computation_columns(self, computation_columns: list[str]):
        self.set_display_columns(computation_columns)

    def num_computation_columns(self) -> int:
        return self.num_display_columns()

    @abstractmethod
    def query(self, query: str) -> pd.DataFrame:
        """Query the DataSourceAbstract
        Args:
            query(str): The SQL query to execute
        """
        pass

    @abstractmethod
    def execute_statement(self, query: str):
        """Execute a non-returning SQL statement
        Args:
            query(str): The SQL query to execute
        """
        pass

    def sample(self, recompute: bool = False) -> pd.DataFrame:
        """Return a sample DataFrame from this DataSource
        Args:
            recompute (bool): Recompute the sample (default: False)
        Returns:
            pd.DataFrame: A sample DataFrame from this DataSource
        """

        # Check if we have a cached sample of rows
        storage_key = f"data_source:{self.uuid}:sample"
        if not recompute and self.data_storage.get(storage_key):
            return pd.read_json(StringIO(self.data_storage.get(storage_key)))

        # No Cache, so we have to compute a sample of data
        self.log.info(f"Sampling {self.uuid}...")
        df = self.sample_impl()
        self.data_storage.set(storage_key, df.to_json())
        return df

    @abstractmethod
    def sample_impl(self) -> pd.DataFrame:
        """Return a sample DataFrame from this DataSourceAbstract
        Returns:
            pd.DataFrame: A sample DataFrame from this DataSource
        """
        pass

    @abstractmethod
    def descriptive_stats(self, recompute: bool = False) -> dict[dict]:
        """Compute Descriptive Stats for all the numeric columns in a DataSource
        Args:
            recompute (bool): Recompute the descriptive stats (default: False)
        Returns:
            dict(dict): A dictionary of descriptive stats for each column in the form
                 {'col1': {'min': 0, 'q1': 1, 'median': 2, 'q3': 3, 'max': 4},
                  'col2': ...}
        """
        pass

    def outliers(self, scale: float = 1.5, recompute: bool = False) -> pd.DataFrame:
        """Return a DataFrame of outliers from this DataSource
        Args:
            scale (float): The scale to use for the IQR (default: 1.5)
            recompute (bool): Recompute the outliers (default: False)
        Returns:
            pd.DataFrame: A DataFrame of outliers from this DataSource
        Notes:
            Uses the IQR * 1.5 (~= 2.5 Sigma) method to compute outliers
            The scale parameter can be adjusted to change the IQR multiplier
        """

        # Check if we have cached outliers
        storage_key = f"data_source:{self.uuid}:outliers"
        if not recompute and self.data_storage.get(storage_key):
            return pd.read_json(StringIO(self.data_storage.get(storage_key)))

        # No Cache, so we have to compute the outliers
        self.log.info(f"Computing Outliers {self.uuid}...")
        df = self.outliers_impl()
        self.data_storage.set(storage_key, df.to_json())
        return df

    @abstractmethod
    def outliers_impl(self, scale: float = 1.5, recompute: bool = False) -> pd.DataFrame:
        """Return a DataFrame of outliers from this DataSource
        Args:
            scale (float): The scale to use for the IQR (default: 1.5)
            recompute (bool): Recompute the outliers (default: False)
        Returns:
            pd.DataFrame: A DataFrame of outliers from this DataSource
        Notes:
            Uses the IQR * 1.5 (~= 2.5 Sigma) method to compute outliers
            The scale parameter can be adjusted to change the IQR multiplier
        """
        pass

    @abstractmethod
    def smart_sample(self) -> pd.DataFrame:
        """Get a SMART sample dataframe from this DataSource
        Returns:
            pd.DataFrame: A combined DataFrame of sample data + outliers
        """
        pass

    @abstractmethod
    def value_counts(self, recompute: bool = False) -> dict[dict]:
        """Compute 'value_counts' for all the string columns in a DataSource
        Args:
            recompute (bool): Recompute the value counts (default: False)
        Returns:
            dict(dict): A dictionary of value counts for each column in the form
                 {'col1': {'value_1': X, 'value_2': Y, 'value_3': Z,...},
                  'col2': ...}
        """
        pass

    @abstractmethod
    def column_stats(self, recompute: bool = False) -> dict[dict]:
        """Compute Column Stats for all the columns in a DataSource
        Args:
            recompute (bool): Recompute the column stats (default: False)
        Returns:
            dict(dict): A dictionary of stats for each column this format
            NB: String columns will NOT have num_zeros and descriptive stats
             {'col1': {'dtype': 'string', 'unique': 4321, 'nulls': 12},
              'col2': {'dtype': 'int', 'unique': 4321, 'nulls': 12, 'num_zeros': 100, 'descriptive_stats': {...}},
              ...}
        """
        pass

    def details(self) -> dict:
        """Additional Details about this DataSourceAbstract Artifact"""
        details = self.summary()
        details["num_rows"] = self.num_rows()
        details["num_columns"] = self.num_columns()
        details["num_display_columns"] = self.num_display_columns()
        details["column_details"] = self.column_details()
        return details

    def expected_meta(self) -> list[str]:
        """DataSources have quite a bit of expected Metadata for EDA displays"""

        # For DataSources, we expect to see the following metadata
        expected_meta = [
            "sageworks_details",
            "sageworks_descriptive_stats",
            "sageworks_value_counts",
            "sageworks_correlations",
            "sageworks_column_stats",
        ]
        return expected_meta

    def ready(self) -> bool:
        """Is the DataSource ready?"""

        # Check if the Artifact is ready
        if not super().ready():
            return False

        # Check if the samples and outliers have been computed
        storage_key = f"data_source:{self.uuid}:sample"
        if not self.data_storage.get(storage_key):
            self.log.important(f"DataSource {self.uuid} doesn't have sample() calling it...")
            self.sample()
        storage_key = f"data_source:{self.uuid}:outliers"
        if not self.data_storage.get(storage_key):
            self.log.important(f"DataSource {self.uuid} doesn't have outliers() calling it...")
            try:
                self.outliers()
            except KeyError:
                self.log.error("DataSource outliers() failed...recomputing columns stats and trying again...")
                self.column_stats(recompute=True)
                self.refresh_meta()
                self.outliers()

        # Okay so we have the samples and outliers, so we are ready
        return True

    def onboard(self) -> bool:
        """This is a BLOCKING method that will onboard the data source (make it ready)

        Returns:
            bool: True if the DataSource was onboarded successfully
        """
        self.log.important(f"Onboarding {self.uuid}...")
        self.set_status("onboarding")
        self.remove_health_tag("needs_onboard")
        self.sample(recompute=True)
        self.column_stats(recompute=True)
        self.refresh_meta()  # Refresh the meta since outliers needs descriptive_stats and value_counts
        self.outliers(recompute=True)

        # Run a health check and refresh the meta
        time.sleep(2)  # Give the AWS Metadata a chance to update
        self.health_check()
        self.refresh_meta()
        self.details(recompute=True)
        self.set_status("ready")
        return True
