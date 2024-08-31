"""CreateView: An abstract base class for View Creation (training, data_quality, etc)"""

import logging
from typing import Union
from abc import ABC, abstractmethod
import awswrangler as wr

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet
from sageworks.core.artifacts.feature_set_core import FeatureSetCore
from sageworks.api import Meta
from sageworks.core.views.view import View


class CreateView(ABC):
    """CreateView: An abstract base class for View Creation (training, data_quality, etc)"""

    # Class attributes
    log = logging.getLogger("sageworks")
    meta = Meta()

    def __init__(self, artifact: Union[DataSource, FeatureSet], view_name: str):
        """CreateView Constructor: Retrieve a CreateView for the given artifact

        Args:
            artifact (Union[DataSource, FeatureSet]): A DataSource or FeatureSet object
            view_name (str): The name of the view to create (e.g. "training")
        """

        # Set the view name
        self.view_name = view_name

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

    def table_name(self) -> str:
        """Construct the view table name for the given view type

        Returns:
            str: The view table name
        """
        if self.view_name == "base":
            return self.base_table
        return f"{self.base_table}_{self.view_name}"

    @abstractmethod
    def create_view(self, source_table: str = None, **kwargs) -> View:
        """Abstract Method: Create the view, each subclass must implement this method

        Args:
            source_table (str, optional): The table/view to create the view from. Defaults to data_source base table.
            **kwargs: Additional keyword arguments that are specific to the view type
        """
        pass

    def delete(self):
        """Delete the database view if it exists."""

        # Log the deletion
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
