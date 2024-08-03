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
from sageworks.core.views import training_view


# Enumerated View Types
class ViewType(Enum):
    """Enumerated Types for SageWorks View Types"""

    RAW = "raw"
    DISPLAY = "display"
    COMPUTATION = "computation"
    TRAINING = "training"
    DATA_QUALITY = "data_quality"


class View:
    def __init__(self, artifact: Union[DataSource, FeatureSet]):
        """View: A View in the Model, View, Controller sense. Provides a view
                (training, data_quality, etc) for DataSources and FeatureSets.

        Args:
            artifact (Union[DataSource, FeatureSet]): A DataSource or FeatureSet object
        """
        self.log = logging.getLogger("sageworks")

        # Is this a DataSource or a FeatureSet?
        self.is_feature_set = isinstance(artifact, FeatureSetCore)

        # Get the data_source from the artifact
        self.data_source = artifact.data_source if self.is_feature_set else artifact
        self.data_source_name = artifact.uuid
        self.database = self.data_source.get_database()
        self.base_table = self.data_source.get_table_name()

    def pull_dataframe(self, view_type: ViewType = ViewType.RAW, limit=50000) -> Union[pd.DataFrame, None]:
        """Pull a DataFrame based on the view type

        Args:
            view_type (ViewType): The type of view to create (default: ViewType.RAW)
            limit (int): The maximum number of rows to pull (default: 50000)

        Returns:
            Union[pd.DataFrame, None]: The DataFrame for the view or None if it doesn't exist
        """
        view_table_name = self.view_table_name(view_type)

        # Check if the view exists
        if not self.exists(view_type):
            self.log.error(f"View {view_table_name} for {self.data_source_name} does not exist.")
            return None

        # Pull the DataFrame
        pull_query = f"SELECT * FROM {view_table_name}"
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

    def create_training_view(self, id_column: str, holdout_ids: Union[list[str], None] = None):
        """Create the view in the database

        Args:
            view_type (ViewType): The type of view to create
        """
        training_view.create_training_view(self.data_source, id_column, holdout_ids)

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
        if self.is_feature_set:
            return f'View: {self.database}:{self.base_table} for FeatureSet("{self.data_source_name}")'
        else:
            return f'View: {self.database}:{self.base_table} for DataSource("{self.data_source_name}")'

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

    # Now let's check if these views exist
    assert my_view.exists(ViewType.RAW) is True
    assert my_view.exists(ViewType.TRAINING) is False

    # Pull the raw data
    df = my_view.pull_dataframe()
    print(df)

    # Create a View for a FeatureSet
    fs = FeatureSet("test_features")
    my_view = View(fs)
    print(my_view)

    # Now let's check if these views exist
    assert my_view.exists(ViewType.RAW) is True

    # Create the training view
    my_view.create_training_view("id")

    # Pull the training data
    df_train = my_view.pull_dataframe(ViewType.TRAINING)
    print(df_train.columns)

    # Delete the training view
    my_view.delete(ViewType.TRAINING)

    # Create a View for the Non-Existing DataSource
    data_source = DataSource("non_existent_data")
    no_data_view = View(data_source)
    print(no_data_view)
    print(no_data_view.exists(ViewType.RAW))

    # Pull the training data (for a view that doesn't exist)
    df_data_quality = my_view.pull_dataframe(ViewType.DATA_QUALITY)
    assert df_data_quality is None
