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
    def __init__(self, artifact: Union[DataSource, FeatureSet], view_type: ViewType = ViewType.RAW):
        """View: A View in the Model, View, Controller sense. Provides a view
                (training, data_quality, etc) for DataSources and FeatureSets.

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
        self.auto_created = False
        if not self.exists(self.view_type):
            self.log.warning(f"View {self.view_type} for {self.data_source_name} does not exist. Creating identity view...")
            self.create_identity_view(self.view_type)

    def pull_dataframe(self, limit:int = 50000, head:bool = False) -> Union[pd.DataFrame, None]:
        """Pull a DataFrame based on the view type

        Args:
            view_type (ViewType): The type of view to create (default: ViewType.RAW)
            limit (int): The maximum number of rows to pull (default: 50000)
            head (bool): Return just the head of the DataFrame (default: False)

        Returns:
            Union[pd.DataFrame, None]: The DataFrame for the view or None if it doesn't exist
        """
        view_table_name = self.view_table_name(self.view_type)

        # Pull the DataFrame
        if head:
            limit = 5
        pull_query = f"SELECT * FROM {view_table_name} LIMIT {limit}"
        df = self.data_source.query(pull_query)
        return df

    def exists(self, view_type: ViewType) -> bool:
        """Check if the view exists in the database

        Args:
            view_type (ViewType): The type of view to check

        Returns:
            bool: True if the view exists, False otherwise.
        """

        # Check for raw view
        if view_type == ViewType.RAW:
            table_name = self.base_table
        else:
            table_name = self.view_table_name(view_type)

        # Query to check if the table/view exists
        check_table_query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = '{self.database}' AND table_name = '{table_name}'
        """
        df = self.data_source.query(check_table_query)
        if not df.empty:
            return True
        else:
            return False

    def auto_create_view(self, view_type: ViewType):
        """Automatically create a view. This is called when a view is accessed that doesn't exist.

        Args:
            view_type (ViewType): The type of view to create
        """
        self.auto_created = True

        # Is this a Training View and do we have an auto id column?
        if view_type == ViewType.TRAINING and self.auto_id_column:
            self.create_training_view(self.auto_id_column)
            return

        # Display View
        if view_type == ViewType.DISPLAY:
            self.create_display_view()

        # Computation View
        if view_type == ViewType.COMPUTATION:
            self.create_computation_view()

        # Okay, so at this point we kind of punt and create an identity view
        self.create_identity_view(view_type)

    def create_training_view(self, id_column: str, holdout_ids: Union[list[str], None] = None):
        """Create a training view for this data source

        Args:
            id_column (str): The name of the id column
            holdout_ids (Union[list[str], None], optional): A list of holdout ids. Defaults to None
        """
        training_view.create_training_view(self.data_source, id_column, holdout_ids)

    def create_display_view(self, column_list: Union[list[str], None] = None, column_limit: int = 30):
        """Create a display view for this data source

        Args:
            column_list (Union[list[str], None], optional): A list of columns to include. Defaults to None.
            column_limit (int, optional): The max number of columns to include. Defaults to 30.
        """
        display_view.create_display_view(self.data_source, column_list, column_limit)

    def create_computation_view(self, column_list: Union[list[str], None] = None, column_limit: int = 30):
        """Create a computation view for this data source

        Args:
            column_list (Union[list[str], None], optional): A list of columns to include. Defaults to None.
            column_limit (int, optional): The max number of columns to include. Defaults to 30.
        """
        display_view.create_display_view(self.data_source, column_list, column_limit)

    def create_identity_view(self, view_type: ViewType):
        """Create a computation view for this data source

        Args:
            column_list (Union[list[str], None], optional): A list of columns to include. Defaults to None.
            column_limit (int, optional): The max number of columns to include. Defaults to 30.
        """
        identity_view.create_display_view(self.data_source, view_type)

    def delete(self, view_type: ViewType):
        """Delete the database view if it exists.

        Args:
            view_type (ViewType): The type of view to delete
        """

        # Construct the view table name
        view_table_name = self.view_table_name(view_type)

        # Check if the view exists
        if not self.exists(view_type):
            self.log.info(f"View {view_table_name} for {self.data_source_name} does not exist, nothing to delete.")
            return

        # If the view exists, drop it
        self.log.important(f"Dropping View {view_table_name}...")
        drop_view_query = f"DROP VIEW {view_table_name}"

        # Execute the DROP VIEW query
        self.data_source.execute_statement(drop_view_query)

    def __repr__(self):
        """Return a string representation of this object"""
        auto = " (Auto-Created)" if self.auto_created else ""
        if self.is_feature_set:
            return f'View: {self.database}:{self.base_table}{auto} for FeatureSet("{self.data_source_name}")'
        else:
            return f'View: {self.database}:{self.base_table}{auto} for DataSource("{self.data_source_name}")'

    # Helper Methods
    def view_table_name(self, view_type: ViewType) -> str:
        """Construct the view table name for the given view type

        Args:
            view_type (ViewType): The given view type

        Returns:
            str: The view table name
        """
        if view_type == ViewType.RAW:
            return self.base_table
        return f"{self.base_table}_{view_type.value}"


if __name__ == "__main__":
    """Exercise the ViewManager Class"""
    from sageworks.api import DataSource, FeatureSet

    # Create a View for the DataSource
    data_source = DataSource("test_data")
    my_view = View(data_source)
    print(my_view)

    # Pull the raw data
    df = my_view.pull_dataframe()
    print(df)

    # Pull just the head
    df_head = my_view.pull_dataframe(head=True)
    print(df_head)

    # Get a training view (that doesn't exist)
    my_view = View(data_source, ViewType.TRAINING)

    # Create a View for a FeatureSet
    fs = FeatureSet("test_features")
    my_view = View(fs)
    print(my_view)

    # Now let's check if these views exist
    assert my_view.exists(ViewType.RAW) is True

    # Create the training view
    my_view.create_training_view("id")

    # Pull the training data
    df_train = my_view.pull_dataframe()
    print(df_train.columns)

    # Delete the training view
    my_view.delete(ViewType.TRAINING)

    # Create a View for the Non-Existing DataSource
    data_source = DataSource("non_existent_data")
    no_data_view = View(data_source)
    print(no_data_view)
    print(no_data_view.exists(ViewType.RAW))

    # Pull the training data (for a view that doesn't exist)
    df_data_quality = my_view.pull_dataframe()
    assert df_data_quality is None
