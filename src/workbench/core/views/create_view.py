"""CreateView: An abstract base class for View Creation (training, display, etc)"""

import logging
from typing import Union
from abc import ABC, abstractmethod

# Workbench Imports
from workbench.api import DataSource, FeatureSet
from workbench.core.artifacts.feature_set_core import FeatureSetCore
from workbench.core.views.view import View


class CreateView(ABC):
    """CreateView: An abstract base class for View Creation (training, display, etc)"""

    # Class attributes
    log = logging.getLogger("workbench")

    def __init__(self, view_name: str, artifact: Union[DataSource, FeatureSet], source_table: str = None):
        """CreateView base class constructor

        Args:
            view_name (str): The name of the view
            artifact (Union[DataSource, FeatureSet]): The DataSource or FeatureSet object
            source_table (str, optional): The table/view to create the view from. Defaults to None
        """
        self.view_name = view_name

        # Is this a DataSource or a FeatureSet?
        self.is_feature_set = isinstance(artifact, FeatureSetCore)
        self.auto_id_column = artifact.id_column if self.is_feature_set else None

        # Set up data source and database details
        self.data_source = artifact.data_source if self.is_feature_set else artifact
        self.database = self.data_source.database

        # Set table names
        self.base_table_name = self.data_source.table
        self.source_table = source_table or self.base_table_name
        self.table = f"{self.base_table_name}___{self.view_name}"

    @abstractmethod
    def create(self, **kwargs) -> Union[View, None]:
        """Abstract Method: Create the view, each subclass must implement this method

        Args:
            **kwargs: Additional keyword arguments specific to the view type

        Returns:
            Union[View, None]: The created View object (or None if failed to create the view)
        """
        pass
