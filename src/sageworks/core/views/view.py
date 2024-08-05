"""View: A View in the Model, View, Controller sense.
         Provides a view (training, data_quality, etc) for DataSources and FeatureSets.
"""

import logging
from enum import Enum
from typing import Union
import pandas as pd

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet
from sageworks.core.artifacts.feature_set_core import FeatureSetCore
from sageworks.core.views import training_view, display_view, computation_view, identity_view


# Enumerated View Types
class ViewType(Enum):
    """Enumerated Types for SageWorks View Types"""

    RAW = "raw"
    DISPLAY = "display"
    COMPUTATION = "computation"
    TRAINING = "training"
    DATA_QUALITY = "data_quality"

    @staticmethod
    def from_string(view_type_str):
        try:
            return ViewType(view_type_str)
        except ValueError:
            raise ValueError(f"Unknown view type: {view_type_str}")


class View:
    """View: A View in the Model, View, Controller sense. Provides a view
            (training, data_quality, etc) for DataSources and FeatureSets.

    Common Usage:
        ```
        view = View(DataSource/FeatureSet, view_type=ViewType.TRAINING)
        training_df = view.pull_dataframe()
        ```
    """

    def __init__(self, artifact: Union[DataSource, FeatureSet], view_type: ViewType = ViewType.RAW):
        """View Constructor: Create a new View object for the given artifact

        Args:
            artifact (Union[DataSource, FeatureSet]): A DataSource or FeatureSet object
            view_type (ViewType, optional): The type of view to create (default: ViewType.RAW)
        """

        self.log = logging.getLogger("sageworks")

        # Is this a DataSource or a FeatureSet?
        self.is_feature_set = isinstance(artifact, FeatureSetCore)
        self.auto_id_column = artifact.record_id if self.is_feature_set else None

        # Get the data_source from the artifact
        self.data_source = artifact.data_source if self.is_feature_set else artifact
        self.data_source_name = artifact.uuid
        self.database = self.data_source.get_database()
        self.base_table = self.data_source.get_table_name()

        # Check if the data source exists
        if not self.data_source.exists():
            raise ValueError(f"Data Source {self.data_source_name} does not exist!")

        # Check if the view exists
        self.view_type = view_type
        self.view_table_name = self.table_name()
        self.auto_created = False
        if not self._exists():
            self.log.warning(f"View {self.view_type} for {self.data_source_name} does not exist. Auto creating view...")
            self._auto_create_view(self.view_type)

        # View Exists so report that we found it
        else:
            self.log.info(f"View {self.view_table_name} for {self.data_source_name} found.")

    def pull_dataframe(self, limit: int = 50000, head: bool = False) -> Union[pd.DataFrame, None]:
        """Pull a DataFrame based on the view type

        Args:
            view_type (ViewType): The type of view to create (default: ViewType.RAW)
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

    def table_name(self) -> str:
        """Construct the view table name for the given view type

        Returns:
            str: The view table name
        """
        if self.view_type == ViewType.RAW:
            return self.base_table
        return f"{self.base_table}_{self.view_type.value}"

    def delete(self):
        """Delete the database view if it exists."""

        # Check if the view exists
        if not self._exists():
            self.log.info(f"View {self.view_table_name} for {self.data_source_name} does not exist, nothing to delete.")
            return

        # If the view exists, drop it
        self.log.important(f"Dropping View {self.view_table_name}...")
        drop_view_query = f"DROP VIEW {self.view_table_name}"

        # Execute the DROP VIEW query
        self.data_source.execute_statement(drop_view_query)

    def _exists(self) -> bool:
        """Internal: Check if the view exists in the database

        Returns:
            bool: True if the view exists, False otherwise.
        """

        # Query to check if the table/view exists
        check_table_query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = '{self.database}' AND table_name = '{self.view_table_name}'
        """
        df = self.data_source.query(check_table_query)
        if not df.empty:
            return True
        else:
            return False

    def _auto_create_view(self, view_type: ViewType):
        """Internal: Automatically create a view. This is called when a view is accessed that doesn't exist.

        Args:
            view_type (ViewType): The type of view to create
        """
        self.auto_created = True

        # Is this a Training View and do we have an auto id column?
        if view_type == ViewType.TRAINING and self.auto_id_column:
            self._create_training_view(self.auto_id_column)
            return

        # Display View
        if view_type == ViewType.DISPLAY:
            self._create_display_view()

        # Computation View
        elif view_type == ViewType.COMPUTATION:
            self._create_computation_view()

        # Okay, so at this point we kind of punt and create an identity view
        else:
            self._create_identity_view(view_type)

    def _create_training_view(self, id_column: str, holdout_ids: Union[list[str], None] = None):
        """Internal: Create a training view for this data source

        Args:
            id_column (str): The name of the id column
            holdout_ids (Union[list[str], None], optional): A list of holdout ids. Defaults to None
        """
        training_view.create_training_view(self.data_source, id_column, holdout_ids)

    def _create_display_view(self, column_list: Union[list[str], None] = None, column_limit: int = 30):
        """Internal: Create a display view for this data source

        Args:
            column_list (Union[list[str], None], optional): A list of columns to include. Defaults to None.
            column_limit (int, optional): The max number of columns to include. Defaults to 30.
        """
        display_view.create_display_view(self.data_source, column_list, column_limit)

    def _create_computation_view(self, column_list: Union[list[str], None] = None, column_limit: int = 30):
        """Internal: Create a computation view for this data source

        Args:
            column_list (Union[list[str], None], optional): A list of columns to include. Defaults to None.
            column_limit (int, optional): The max number of columns to include. Defaults to 30.
        """
        computation_view.create_computation_view(self.data_source, column_list, column_limit)

    def _create_identity_view(self, view_type: ViewType):
        """Internal: Create a computation view for this data source

        Args:
            column_list (Union[list[str], None], optional): A list of columns to include. Defaults to None.
            column_limit (int, optional): The max number of columns to include. Defaults to 30.
        """
        identity_view.create_display_view(self.data_source, view_type)

    def __repr__(self):
        """Return a string representation of this object"""
        auto = " (Auto-Created)" if self.auto_created else ""
        if self.is_feature_set:
            return f'View: {self.database}:{self.view_table_name}{auto} for FeatureSet("{self.data_source_name}")'
        else:
            return f'View: {self.database}:{self.view_table_name}{auto} for DataSource("{self.data_source_name}")'


if __name__ == "__main__":
    """Exercise the ViewManager Class"""
    from sageworks.api import DataSource, FeatureSet

    # Create a View for the DataSource
    data_source = DataSource("test_data")
    raw_view = View(data_source)  # Default is RAW
    print(raw_view)

    # Pull the raw data
    df = raw_view.pull_dataframe()
    print(df)

    # Create a display view
    my_display_view = View(data_source, ViewType.DISPLAY)

    # Pull just the head
    df_head = my_display_view.pull_dataframe(head=True)
    print(df_head)

    # Create a View for a FeatureSet
    fs = FeatureSet("test_features")
    my_view = View(fs, ViewType.TRAINING)

    # Pull the training data
    df_train = my_view.pull_dataframe()
    print(df_train.columns)

    # Delete the training view
    my_view.delete()

    # Create a View for the Non-Existing DataSource
    data_source = DataSource("non_existent_data")
    try:
        no_data_view = View(data_source)
    except ValueError:
        print("Expected Error == Good :)")
