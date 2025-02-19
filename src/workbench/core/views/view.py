"""View: Read from a view (training, display, etc) for DataSources and FeatureSets."""

import logging
import time
from typing import Union
import pandas as pd
import awswrangler as wr

# Workbench Imports
from workbench.api import DataSource, FeatureSet, Meta
from workbench.core.artifacts.feature_set_core import FeatureSetCore
from workbench.core.views.view_utils import list_supplemental_data_tables, delete_table, view_details


class View:
    """View: Read from a view (training, display, etc) for DataSources and FeatureSets.

    Common Usage:
        ```python

        # Grab the Display View for a DataSource
        display_view = ds.view("display")
        print(display_view.columns)

        # Pull a DataFrame for the view
        df = display_view.pull_dataframe()

        # Views also work with FeatureSets
        comp_view = fs.view("computation")
        comp_df = comp_view.pull_dataframe()

        # Query the view with a custom SQL query
        query = f"SELECT * FROM {comp_view.table} WHERE age > 30"
        df = comp_view.query(query)

        # Delete the view
        comp_view.delete()
        ```
    """

    # Class attributes
    log = logging.getLogger("workbench")
    meta = Meta()

    def __init__(self, artifact: Union[DataSource, FeatureSet], view_name: str, **kwargs):
        """View Constructor: Retrieve a View for the given artifact

        Args:
            artifact (Union[DataSource, FeatureSet]): A DataSource or FeatureSet object
            view_name (str): The name of the view to retrieve (e.g. "training")
        """

        # Set the view name
        self.view_name = view_name

        # Is this a DataSource or a FeatureSet?
        self.is_feature_set = isinstance(artifact, FeatureSetCore)
        self.auto_id_column = artifact.id_column if self.is_feature_set else None

        # Get the data_source from the artifact
        self.artifact_name = artifact.uuid
        self.data_source = artifact.data_source if self.is_feature_set else artifact
        self.database = self.data_source.database

        # Construct our base_table_name
        self.base_table_name = self.data_source.table

        # Check if the view should be auto created
        self.auto_created = False
        if kwargs.get("auto_create_view", True) and not self.exists():

            # A direct double check before we auto-create
            if not self.exists(skip_cache=True):
                self.log.important(
                    f"View {self.view_name} for {self.artifact_name} doesn't exist, attempting to auto-create..."
                )
                self.auto_created = self._auto_create_view()

                # Check for failure of the auto-creation
                if not self.auto_created:
                    self.log.error(
                        f"View {self.view_name} for {self.artifact_name} doesn't exist and cannot be auto-created..."
                    )
                    self.view_name = self.columns = self.column_types = self.source_table = self.base_table_name = (
                        self.join_view
                    ) = None
                    return

        # Now fill some details about the view
        self.columns, self.column_types, self.source_table, self.join_view = view_details(
            self.table, self.data_source.database, self.data_source.boto3_session
        )

    def pull_dataframe(self, limit: int = 50000) -> Union[pd.DataFrame, None]:
        """Pull a DataFrame based on the view type

        Args:
            limit (int): The maximum number of rows to pull (default: 50000)

        Returns:
            Union[pd.DataFrame, None]: The DataFrame for the view or None if it doesn't exist
        """

        # Pull the DataFrame
        pull_query = f'SELECT * FROM "{self.table}" LIMIT {limit}'
        df = self.data_source.query(pull_query)
        return df

    def query(self, query: str) -> Union[pd.DataFrame, None]:
        """Query the view with a custom SQL query

        Args:
            query (str): The SQL query to execute

        Returns:
            Union[pd.DataFrame, None]: The DataFrame for the query or None if it doesn't exist
        """
        return self.data_source.query(query)

    def column_details(self) -> dict:
        """Return a dictionary of the column names and types for this view

        Returns:
            dict: A dictionary of the column names and types
        """
        return dict(zip(self.columns, self.column_types))

    @property
    def table(self) -> str:
        """Construct the view table name for the given view type

        Returns:
            str: The view table name
        """
        if self.view_name is None:
            return None
        if self.view_name == "base":
            return self.base_table_name
        return f"{self.base_table_name}___{self.view_name}"

    def delete(self):
        """Delete the database view (and supplemental data) if it exists."""

        # List any supplemental tables for this data source
        supplemental_tables = list_supplemental_data_tables(self.base_table_name, self.database)
        for table in supplemental_tables:
            if self.view_name in table:
                self.log.important(f"Deleting Supplemental Table {table}...")
                delete_table(table, self.database, self.data_source.boto3_session)

        # Now drop the view
        self.log.important(f"Dropping View {self.table}...")
        drop_view_query = f'DROP VIEW "{self.table}"'

        # Execute the DROP VIEW query
        try:
            self.data_source.execute_statement(drop_view_query, silence_errors=True)
        except wr.exceptions.QueryFailed as e:
            if "View not found" in str(e):
                self.log.info(f"View {self.table} not found, this is fine...")
            else:
                raise

        # We want to do a small sleep so that AWS has time to catch up
        self.log.info("Sleeping for 3 seconds after dropping view to allow AWS to catch up...")
        time.sleep(3)

    def exists(self, skip_cache: bool = False) -> bool:
        """Check if the view exists in the database

        Args:
            skip_cache (bool): Skip the cache and check the database directly (default: False)
        Returns:
            bool: True if the view exists, False otherwise.
        """
        # The BaseView always exists
        if self.view_name == "base":
            return True

        # If we're skipping the cache, we need to check the database directly
        if skip_cache:
            return self._check_database()

        # Use the meta class to see if the view exists
        views_df = self.meta.views(self.database)

        # Check if we have ANY views
        if views_df.empty:
            return False

        # Check if the view exists
        return self.table in views_df["Name"].values

    def ensure_exists(self):
        """Ensure if the view exists by making a query directly to the database. If it doesn't exist, create it"""

        # The BaseView always exists
        if self.view_name == "base":
            return True

        # Check the database directly
        if not self._check_database():
            self._auto_create_view()

    def _check_database(self) -> bool:
        """Internal: Check if the view exists in the database

        Returns:
            bool: True if the view exists, False otherwise
        """
        # Query to check if the table/view exists
        check_table_query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = '{self.database}' AND table_name = '{self.table}'
        """
        _df = self.data_source.query(check_table_query)
        return not _df.empty

    def _auto_create_view(self) -> bool:
        """Internal: Automatically create a view training, display, and computation views

        Returns:
            bool: True if the view was created, False otherwise

        Raises:
            ValueError: If the view type is not supported
        """
        from workbench.core.views import DisplayView, ComputationView, TrainingView

        # First if we're going to auto-create, we need to make sure the data source exists
        if not self.data_source.exists():
            self.log.error(f"Data Source {self.data_source.uuid} does not exist...")
            return False

        # DisplayView
        if self.view_name == "display":
            self.log.important(f"Auto creating View {self.view_name} for {self.data_source.uuid}...")
            DisplayView.create(self.data_source)
            return True

        # ComputationView
        if self.view_name == "computation":
            self.log.important(f"Auto creating View {self.view_name} for {self.data_source.uuid}...")
            ComputationView.create(self.data_source)
            return True

        # TrainingView
        if self.view_name == "training":
            # We're only going to create training views for FeatureSets
            if self.is_feature_set:
                self.log.important(f"Auto creating View {self.view_name} for {self.data_source.uuid}...")
                TrainingView.create(self.data_source, id_column=self.auto_id_column)
                return True
            else:
                self.log.warning("Training Views are only supported for FeatureSets...")
                return False

        # If we get here, we don't support auto-creating this view
        self.log.warning(f"Auto-Create for {self.view_name} not implemented yet...")
        return False

    def __repr__(self):
        """Return a string representation of this object"""

        # Set up various details that we want to print out
        auto = "(Auto-Created)" if self.auto_created else ""
        artifact = "FeatureSet" if self.is_feature_set else "DataSource"

        info = f'View: "{self.view_name}" for {artifact}("{self.artifact_name}")\n'
        info += f"      Database: {self.database}\n"
        info += f"      Table: {self.table}{auto}\n"
        info += f"      Source Table: {self.source_table}\n"
        info += f"      Join View: {self.join_view}"
        return info


if __name__ == "__main__":
    """Exercise the ViewManager Class"""
    # See tests/views/views_tests.py for more examples
    import numpy as np
    from workbench.core.views.pandas_to_view import PandasToView

    # Show trace calls
    logging.getLogger("workbench").setLevel(logging.DEBUG)

    # Grab the Display View for a DataSource
    data_source = DataSource("abalone_data")
    display_view = View(data_source, "display")
    print(display_view)

    # Pull the raw data
    df = display_view.pull_dataframe()
    print(df)

    # Grab a Computation View for a DataSource (that doesn't exist)
    computation_view = View(data_source, "computation")
    print(computation_view)
    print(computation_view.columns)

    # Delete the computation view
    computation_view.delete()

    # Create a display View for a FeatureSet
    fs = FeatureSet("test_features")
    display_view = View(fs, "display")
    df_head = display_view.pull_dataframe(limit=5)
    print(df_head)

    # Pull the columns for the display view
    print(display_view.columns)

    # Delete the display view
    display_view.delete()

    # Generate a DataFrame with the same id column and two random columns
    my_df = fs.pull_dataframe()[["id", "height", "weight", "salary"]]
    my_df["random1"] = np.random.rand(len(my_df))
    my_df["random2"] = np.random.rand(len(my_df))

    # Create a view with a PandasToView class
    df_view = PandasToView.create("test_view", fs, df=my_df, id_column="id")

    # Test supplemental data tables deletion
    view = View(fs, "test_view")
    view.delete()
