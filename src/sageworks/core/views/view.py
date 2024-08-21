"""View: A View in the Model, View, Controller sense.
         Provides a view (training, data_quality, etc) for DataSources and FeatureSets.
"""

import logging
from enum import Enum
from typing import Union
import pandas as pd
from abc import ABC, abstractmethod
import importlib
import awswrangler as wr

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet
from sageworks.core.artifacts.feature_set_core import FeatureSetCore
from sageworks.api import Meta


# Enumerated View Types
class ViewType(Enum):
    """Enumerated Types for SageWorks View Types"""

    BASE = "base"
    IDENTITY = "identity"
    DISPLAY = "display"
    COMPUTATION = "computation"
    TRAINING = "training"
    DATA_QUALITY = "data_quality"

    @staticmethod
    def from_string(view_type_str: str) -> "ViewType":
        """Convert a string to a ViewType Enum"""
        try:
            return ViewType(view_type_str)
        except ValueError:
            raise ValueError(f"Unknown view type: {view_type_str}")


class View(ABC):
    """View: A View in the Model, View, Controller sense. Provides a view
            (training, data_quality, etc) for DataSources and FeatureSets.

    Common Usage:
        ```
        view = View(DataSource/FeatureSet, "training")
        training_df = view.pull_dataframe()
        ```
    """

    # Class attributes
    log = logging.getLogger("sageworks")
    meta = Meta()

    @classmethod
    def factory(cls, artifact: Union[DataSource, FeatureSet], view_type: str = "base"):
        # Convert the view_type string to a ViewType Enum
        view_type_enum = ViewType.from_string(view_type)

        # Dynamically get the subclass module and class name
        module_name, class_name = VIEW_SUBCLASS_MAP[view_type_enum].rsplit(".", 1)
        module = importlib.import_module(module_name)
        subclass = getattr(module, class_name)

        # Create and return the subclass instance
        return subclass(artifact)

    def __init__(self, artifact: Union[DataSource, FeatureSet], view_type: ViewType):
        """View Constructor: Create a new View object for the given artifact

        Args:
            artifact (Union[DataSource, FeatureSet]): A DataSource or FeatureSet object
            view_type (ViewType): The type of view to create (e.g. ViewType.TRAINING)
        """

        # Set the view type
        self.view_type = view_type

        # Is this a DataSource or a FeatureSet?
        self.is_feature_set = isinstance(artifact, FeatureSetCore)
        self.auto_id_column = artifact.record_id if self.is_feature_set else None

        # Get the data_source from the artifact
        self.data_source = artifact.data_source if self.is_feature_set else artifact
        self.data_source_name = artifact.uuid
        self.database = self.data_source.get_database()

        # Set our view table name
        self.base_table = self.data_source.get_table_name()
        self.view_table_name = self.table_name()

        # Check if the view exists
        self.auto_created = False
        if not self.exists():
            self.log.important(
                f"View {self.view_type} for {self.data_source_name} does not exist. Auto creating view..."
            )
            self._auto_create_view()

        # View Exists so report that we found it
        else:
            self.log.info(f"View {self.view_table_name} for {self.data_source_name} found.")

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
        pull_query = f"SELECT * FROM {self.view_table_name} LIMIT {limit}"
        df = self.data_source.query(pull_query)
        return df

    def columns(self) -> list:
        """Return the columns for the view"""

        # Retrieve the table metadata
        glue_client = self.data_source.boto_session.client("glue")
        response = glue_client.get_table(DatabaseName=self.database, Name=self.view_table_name)

        # Extract the column names from the schema
        column_names = [col["Name"] for col in response["Table"]["StorageDescriptor"]["Columns"]]

        # Sanity check the column names
        if not column_names:
            raise ValueError(f"No columns found for view {self.view_table_name}")

        return column_names

    def table_name(self) -> str:
        """Construct the view table name for the given view type

        Returns:
            str: The view table name
        """
        if self.view_type == ViewType.BASE:
            return self.base_table
        return f"{self.base_table}_{self.view_type.value}"

    @abstractmethod
    def create_view(self, source_table: str = None, **kwargs):
        """Abstract Method: Create the view, each subclass must implement this method

        Args:
            source_table (str, optional): The table/view to create the view from. Defaults to data_source base table.
            **kwargs: Additional keyword arguments that are specific to the view type
        """
        pass

    def delete(self):
        """Delete the database view if it exists."""

        # Check if the view exists
        if not self.exists():
            self.log.info(f"View {self.view_table_name} for {self.data_source_name} does not exist, nothing to delete.")
            return

        # If the view exists, drop it
        self.log.important(f"Dropping View {self.view_table_name}...")
        drop_view_query = f"DROP VIEW {self.view_table_name}"

        # Execute the DROP VIEW query
        try:
            self.data_source.execute_statement(drop_view_query, silence_errors=True)
        except wr.exceptions.QueryFailed as e:
            if "View not found" in str(e):
                self.log.info(f"View {self.view_table_name} not found, this is fine...")
            else:
                raise

    def exists(self) -> bool:
        """Check if the view exists in the database

        Returns:
            bool: True if the view exists, False otherwise.
        """
        # The BaseView always exists
        if self.view_type == ViewType.BASE:
            return True

        # Use the meta class to see if the view exists
        views_df = self.meta.views(self.database)
        return self.view_table_name in views_df["Name"].values

    def ensure_exists(self):
        """Ensure if the view exists by making a query directly to the database. If it doesn't exist, create it"""
        # The BaseView always exists
        if self.view_type == ViewType.BASE:
            return True

        # Query to check if the table/view exists
        check_table_query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = '{self.database}' AND table_name = '{self.view_table_name}'
        """
        _df = self.data_source.query(check_table_query)
        if _df.empty:
            self.log.warning(
                f"View {self.view_table_name} for {self.data_source_name} does not exist. Auto creating view..."
            )
            self._auto_create_view()

    def _auto_create_view(self):
        """Internal: Automatically create a view. This is called when a view is accessed that doesn't exist"""
        self.auto_created = True
        try:
            self.create_view()
        except TypeError:
            self.log.important(f"View {self.view_type} for {self.data_source_name} cannot be auto-created.")

    def __repr__(self):
        """Return a string representation of this object"""
        auto = " (Auto-Created)" if self.auto_created else ""
        if self.is_feature_set:
            return f'View: {self.database}:{self.view_table_name}{auto} for FeatureSet("{self.data_source_name}")'
        else:
            return f'View: {self.database}:{self.view_table_name}{auto} for DataSource("{self.data_source_name}")'


# This maps the ViewType Enum to the actual View module.Class (for Factory Logic)
VIEW_SUBCLASS_MAP = {
    ViewType.BASE: "base_view.BaseView",
    ViewType.IDENTITY: "identity_view.IdentityView",
    ViewType.DISPLAY: "display_view.DisplayView",
    ViewType.COMPUTATION: "computation_view.ComputationView",
    ViewType.TRAINING: "training_view.TrainingView",
    ViewType.DATA_QUALITY: "data_quality_view.DataQualityView",
}

if __name__ == "__main__":
    """Exercise the ViewManager Class"""
    from sageworks.api import DataSource, FeatureSet

    # Show trace calls
    logging.getLogger("sageworks").setLevel(logging.DEBUG)

    # Create a Display View for a DataSource
    data_source = DataSource("abalone_data")
    display_view = View.factory(data_source, "display")
    print(display_view)

    # Test direct Class Instantiation
    from sageworks.core.views.base_view import BaseView

    base_view = BaseView(data_source)
    print(base_view)

    # Pull the raw data
    df = base_view.pull_dataframe()
    print(df)

    # Create a display View for a FeatureSet
    fs = FeatureSet("test_features")
    display_view = View.factory(fs, "display")
    df_head = display_view.pull_dataframe(head=True)
    print(df_head)

    # Pull the columns for the display view
    print(display_view.columns())

    # Delete the display view and then call ensure_exists
    display_view.delete()
    display_view.ensure_exists()

    """
    # Create a Training View for a FeatureSet
    fs = FeatureSet("test_features")
    my_view = View(fs, ViewType.TRAINING)

    # Pull the training data
    df_train = my_view.pull_dataframe()
    print(df_train["training"].value_counts())

    # Now create a Training View with holdout ids
    holdout_ids = [1, 2, 3]
    my_view.set_training_holdouts("id", holdout_ids)

    # Pull the training data
    df_train = my_view.pull_dataframe()
    print(df_train["training"].value_counts())

    # Delete the training view
    my_view.delete()

    # Okay now create a training view FROM the computation view
    my_view = View(fs, ViewType.COMPUTATION)
    computation_table = my_view.table_name()
    my_view.set_training_holdouts("id", holdout_ids, source_table=computation_table)

    # Create a View for the Non-Existing DataSource
    data_source = DataSource("non_existent_data")
    try:
        no_data_view = View(data_source)
    except ValueError:
        print("Expected Error == Good :)")
    """
