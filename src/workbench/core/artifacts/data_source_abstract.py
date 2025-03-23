"""DataSourceAbstract: Abstract Base Class for all data sources (S3: CSV, JSONL, Parquet, RDS, etc)"""

from abc import abstractmethod
import pandas as pd
import time

# Workbench Imports
from workbench.core.artifacts.artifact import Artifact
from workbench.utils.deprecated_utils import deprecated

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from workbench.core.views import View


class DataSourceAbstract(Artifact):
    def __init__(self, data_uuid: str, database: str = "workbench", **kwargs):
        """DataSourceAbstract: Abstract Base Class for all data sources
        Args:
            data_uuid(str): The UUID for this Data Source
            database(str): The database to use for this Data Source (default: workbench)
        """

        # Call superclass init
        super().__init__(data_uuid, **kwargs)

        # Set up our instance attributes
        self._database = database
        self._table_name = data_uuid

    def __post_init__(self):
        # Call superclass post_init
        super().__post_init__()

    @deprecated(version="0.9")
    def get_database(self) -> str:
        """Get the database for this Data Source"""
        return self._database

    @property
    def database(self) -> str:
        """Get the database for this Data Source"""
        return self._database

    @property
    def table(self) -> str:
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

    @property
    @abstractmethod
    def columns(self) -> list[str]:
        """Return the column names for this Data Source"""
        pass

    @property
    @abstractmethod
    def column_types(self) -> list[str]:
        """Return the column types for this Data Source"""
        pass

    def column_details(self) -> dict:
        """Return the column details for this Data Source

        Returns:
            dict: The column details for this Data Source
        """
        return dict(zip(self.columns, self.column_types))

    def views(self) -> list[str]:
        """Return the views for this Data Source"""
        from workbench.core.views.view_utils import list_views

        return list_views(self)

    def supplemental_data(self) -> list[str]:
        """Return the supplemental data for this Data Source"""
        from workbench.core.views.view_utils import list_supplemental_data

        return list_supplemental_data(self)

    def view(self, view_name: str) -> "View":
        """Return a DataFrame for a specific view
        Args:
            view_name (str): The name of the view to return
        Returns:
            pd.DataFrame: A DataFrame for the specified view
        """
        from workbench.core.views import View

        return View(self, view_name)

    def set_display_columns(self, diplay_columns: list[str]):
        """Set the display columns for this Data Source

        Args:
            diplay_columns (list[str]): The display columns for this Data Source
        """
        # Check mismatch of display columns to computation columns
        c_view = self.view("computation")
        computation_columns = c_view.columns
        mismatch_columns = [col for col in diplay_columns if col not in computation_columns]
        if mismatch_columns:
            self.log.monitor(f"Display View/Computation mismatch: {mismatch_columns}")

        self.log.important(f"Setting Display Columns...{diplay_columns}")
        from workbench.core.views import DisplayView

        # Create a NEW display view
        DisplayView.create(self, source_table=c_view.table, column_list=diplay_columns)

    def set_computation_columns(self, computation_columns: list[str]):
        """Set the computation columns for this Data Source

        Args:
            computation_columns (list[str]): The computation columns for this Data Source
        """
        self.log.important(f"Setting Computation Columns...{computation_columns}")
        from workbench.core.views import ComputationView

        # Create a NEW computation view
        ComputationView.create(self, column_list=computation_columns)
        self.recompute_stats()

    def _create_display_view(self):
        """Internal: Create the Display View for this DataSource"""
        from workbench.core.views import View

        View(self, "display")

    @abstractmethod
    def query(self, query: str) -> pd.DataFrame:
        """Query the DataSourceAbstract
        Args:
            query(str): The SQL query to execute
        """
        pass

    @abstractmethod
    def execute_statement(self, query: str):
        """Execute an SQL statement that doesn't return a result
        Args:
            query(str): The SQL statement to execute
        """
        pass

    @abstractmethod
    def sample(self) -> pd.DataFrame:
        """Return a sample DataFrame from this DataSourceAbstract

        Returns:
            pd.DataFrame: A sample DataFrame from this DataSource
        """
        pass

    @abstractmethod
    def descriptive_stats(self) -> dict[dict]:
        """Compute Descriptive Stats for all the numeric columns in a DataSource

        Returns:
            dict(dict): A dictionary of descriptive stats for each column in the form
                 {'col1': {'min': 0, 'q1': 1, 'median': 2, 'q3': 3, 'max': 4},
                  'col2': ...}
        """
        pass

    @abstractmethod
    def outliers(self, scale: float = 1.5) -> pd.DataFrame:
        """Return a DataFrame of outliers from this DataSource

        Args:
            scale (float): The scale to use for the IQR (default: 1.5)

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
    def value_counts(self) -> dict[dict]:
        """Compute 'value_counts' for all the string columns in a DataSource

        Returns:
            dict(dict): A dictionary of value counts for each column in the form
                 {'col1': {'value_1': X, 'value_2': Y, 'value_3': Z,...},
                  'col2': ...}
        """
        pass

    @abstractmethod
    def column_stats(self) -> dict[dict]:
        """Compute Column Stats for all the columns in a DataSource

        Returns:
            dict(dict): A dictionary of stats for each column this format
            NB: String columns will NOT have num_zeros and descriptive stats
                {'col1': {'dtype': 'string', 'unique': 4321, 'nulls': 12},
                 'col2': {'dtype': 'int', 'unique': 4321, 'nulls': 12, 'num_zeros': 100, 'descriptive_stats': {...}},
                 ...}
        """
        pass

    @abstractmethod
    def correlations(self) -> dict[dict]:
        """Compute Correlations for all the numeric columns in a DataSource

        Returns:
            dict(dict): A dictionary of correlations for each column in this format
                 {'col1': {'col2': 0.5, 'col3': 0.9, 'col4': 0.4, ...},
                  'col2': {'col1': 0.5, 'col3': 0.8, 'col4': 0.3, ...}}
        """
        pass

    def details(self) -> dict:
        """Additional Details about this DataSourceAbstract Artifact"""
        details = self.summary()
        details["num_rows"] = self.num_rows()
        details["num_columns"] = self.num_columns()
        details["column_details"] = self.column_details()
        return details

    def expected_meta(self) -> list[str]:
        """DataSources have quite a bit of expected Metadata for EDA displays"""

        # For DataSources, we expect to see the following metadata
        expected_meta = [
            # FIXME: Revisit this
            # "workbench_details",
            "workbench_descriptive_stats",
            "workbench_value_counts",
            "workbench_correlations",
            "workbench_column_stats",
        ]
        return expected_meta

    def ready(self) -> bool:
        """Is the DataSource ready?"""

        # Check if the Artifact is ready
        if not super().ready():
            return False

        # If we don't have a smart_sample we're probably not ready
        if not self.df_cache.check(f"{self.uuid}/smart_sample"):
            self.log.warning(f"DataSource {self.uuid} not ready...")
            return False

        # Okay so we have sample, outliers, and smart_sample so we are ready
        return True

    def onboard(self) -> bool:
        """This is a BLOCKING method that will onboard the data source (make it ready)

        Returns:
            bool: True if the DataSource was onboarded successfully
        """
        self.log.important(f"Onboarding {self.uuid}...")
        self.set_status("onboarding")
        self.remove_health_tag("needs_onboard")

        # Make sure our display view actually exists
        self.view("display").ensure_exists()

        # Recompute the stats
        self.recompute_stats()

        # Run a health check and refresh the meta
        time.sleep(2)  # Give the AWS Metadata a chance to update
        self.health_check()
        self.refresh_meta()
        self.details()
        self.set_status("ready")
        return True

    def recompute_stats(self) -> bool:
        """This is a BLOCKING method that will recompute the stats for the data source

        Returns:
            bool: True if the DataSource stats were recomputed successfully
        """
        self.log.important(f"Recomputing Stats {self.uuid}...")

        # Make sure our computation view actually exists
        self.view("computation").ensure_exists()

        # Compute the sample, column stats, outliers, and smart_sample
        self.df_cache.delete(f"{self.uuid}/sample")
        self.sample()
        self.column_stats()
        self.refresh_meta()  # Refresh the meta since outliers needs descriptive_stats and value_counts
        self.df_cache.delete(f"{self.uuid}/outliers")
        self.outliers()
        self.df_cache.delete(f"{self.uuid}/smart_sample")
        self.smart_sample()
        return True
