"""View: Read from a view (training, display, etc) for DataSources and FeatureSets."""

import logging
import time
import re
from typing import Union
import pandas as pd
import awswrangler as wr
from botocore.exceptions import ClientError

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet
from sageworks.core.artifacts.feature_set_core import FeatureSetCore
from sageworks.api import Meta
from sageworks.core.views.view_utils import list_supplemental_data_tables, delete_table, view_details


class View:
    """View: Read from a view (training, display, etc) for DataSources and FeatureSets.

    Common Usage:
        ```
        view = View(DataSource/FeatureSet, "training")
        training_df = view.pull_dataframe()
        ```
    """

    # Class attributes
    log = logging.getLogger("sageworks")
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
        self.auto_id_column = artifact.record_id if self.is_feature_set else None

        # Get the data_source from the artifact
        self.artifact_name = artifact.uuid
        self.data_source = artifact.data_source if self.is_feature_set else artifact
        self.database = self.data_source.get_database()

        # Construct our base_table_name
        self.base_table_name = self.data_source.table_name

        # Check if the view should be auto created
        self.auto_created = False
        if kwargs.get("auto_create_view", True) and not self.exists():
            if self.view_name in ["training", "display", "computation"]:
                self._auto_create_view()
                self.auto_created = True
            else:
                self.log.error(f"View {self.view_name} for {self.data_source.uuid} does not exist...")
                return

        # Now fill some details about the view
        self.columns, self.column_types, self.source_table = view_details(self.data_source.get_database(),
                                                                          self.table_name,
                                                                          self.data_source.boto3_session)

    def pull_dataframe(self, limit: int = 50000, head: bool = False) -> Union[pd.DataFrame, None]:
        """Pull a DataFrame based on the view type

        Args:
            limit (int): The maximum number of rows to pull (default: 50000)
            head (bool): Return just the head of the DataFrame (default: False)

        Returns:
            Union[pd.DataFrame, None]: The DataFrame for the view or None if it doesn't exist
        """

        # Pull the DataFrame
        if head:
            limit = 5
        pull_query = f"SELECT * FROM {self.table_name} LIMIT {limit}"
        df = self.data_source.query(pull_query)
        return df

    def column_details(self) -> dict:
        """Return a dictionary of the column names and types for this view

        Returns:
            dict: A dictionary of the column names and types
        """
        return dict(zip(self.columns, self.column_types))

    @property
    def table_name(self) -> str:
        """Construct the view table name for the given view type

        Returns:
            str: The view table name
        """
        if self.view_name == "base":
            return self.base_table_name
        return f"{self.base_table_name}_{self.view_name}"

    def delete(self):
        """Delete the database view (and supplemental data) if it exists."""

        # List any supplemental tables for this data source
        supplemental_tables = list_supplemental_data_tables(self.data_source)
        for table in supplemental_tables:
            if self.view_name in table:
                self.log.important(f"Deleting Supplemental Table {table}...")
                delete_table(self.data_source, table)

        # Now drop the view
        self.log.important(f"Dropping View {self.table_name}...")
        drop_view_query = f"DROP VIEW {self.table_name}"

        # Execute the DROP VIEW query
        try:
            self.data_source.execute_statement(drop_view_query, silence_errors=True)
        except wr.exceptions.QueryFailed as e:
            if "View not found" in str(e):
                self.log.info(f"View {self.table_name} not found, this is fine...")
            else:
                raise

        # We want to do a small sleep so that AWS has time to catch up
        self.log.info("Sleeping for 3 seconds after dropping view to allow AWS to catch up...")
        time.sleep(3)

    def exists(self) -> bool:
        """Check if the view exists in the database

        Returns:
            bool: True if the view exists, False otherwise.
        """
        # The BaseView always exists
        if self.view_name == "base":
            return True

        # Use the meta class to see if the view exists
        views_df = self.meta.views(self.database)

        # Check if we have ANY views
        if views_df.empty:
            return False

        # Check if the view exists
        return self.table_name in views_df["Name"].values

    def ensure_exists(self):
        """Ensure if the view exists by making a query directly to the database. If it doesn't exist, create it"""

        # The BaseView always exists
        if self.view_name == "base":
            return True

        # Query to check if the table/view exists
        check_table_query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = '{self.database}' AND table_name = '{self.table_name}'
        """
        _df = self.data_source.query(check_table_query)
        if _df.empty:
            self._auto_create_view()

    def _auto_create_view(self):
        """Internal: Automatically create a view training, display, and computation views"""
        from sageworks.core.views import DisplayView, ComputationView, TrainingView

        # First if we're going to auto-create, we need to make sure the data source exists
        if not self.data_source.exists():
            self.log.error(f"Data Source {self.data_source.uuid} does not exist...")
            return

        # Auto create the standard views
        if self.view_name == "display":
            self.log.important(f"Auto creating View {self.view_name} for {self.data_source.uuid}...")
            DisplayView(self.data_source).create()
        elif self.view_name == "computation":
            self.log.important(f"Auto creating View {self.view_name} for {self.data_source.uuid}...")
            ComputationView(self.data_source).create()
        elif self.view_name == "training":
            self.log.important(f"Auto creating View {self.view_name} for {self.data_source.uuid}...")
            TrainingView(self.data_source).create(id_column=self.auto_id_column)
        else:
            self.log.error(f"Auto-Create for {self.view_name} not implemented yet...")

    def __repr__(self):
        """Return a string representation of this object"""

        # Set up various details that we want to print out
        auto = "(Auto-Created)" if self.auto_created else ""
        artifact = "FeatureSet" if self.is_feature_set else "DataSource"

        info = f'View: "{self.view_name}" for {artifact}("{self.artifact_name}")\n'
        info += f"      Database: {self.database}\n"
        info += f"      Table: {self.table_name}{auto}\n"
        info += f"      Source Table: {self.source_table}"
        return info


if __name__ == "__main__":
    """Exercise the ViewManager Class"""
    # See tests/views/views_tests.py for more examples
    import numpy as np
    from sageworks.api import DataSource, FeatureSet
    from sageworks.core.views.create_view_with_df import CreateViewWithDF

    # Show trace calls
    logging.getLogger("sageworks").setLevel(logging.DEBUG)

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
    df_head = display_view.pull_dataframe(head=True)
    print(df_head)

    # Pull the columns for the display view
    print(display_view.columns)

    # Delete the display view
    display_view.delete()

    # Generate a DataFrame with the same id column and two random columns
    my_df = fs.pull_dataframe()[["id", "height", "weight", "salary"]]
    my_df["random1"] = np.random.rand(len(my_df))
    my_df["random2"] = np.random.rand(len(my_df))

    # Create a view with a CreateViewWithDF class
    df_view = CreateViewWithDF("test_df", fs).create(df=my_df, id_column="id")

    # Test supplemental data tables deletion
    view = View(fs, "test_df")
    view.delete()
