"""ViewManager: Used by DataSource to manage 'views' of the data
   Note: This is probably Athena specific, so we may need to refactor this
"""

import logging
from enum import Enum


# Enumerated View Types
class ViewType(Enum):
    """Enumerated Types for SageWorks View Types"""

    BASE = "base"
    DISPLAY = "display"
    COMPUTATION = "computation"
    TRAINING = "training"


class View:
    def __init__(self, name: str, data_source):
        """View: A View Class to be used by the ViewManager
        Args:
            name(str): The name of the view
            data_source: The DataSource that this view is created from
        """

        # Set up our instance attributes
        self.log = logging.getLogger("sageworks")
        self.name = name
        self.data_source = data_source
        self.database = data_source.get_database()
        self.base_table = data_source.get_table_name()
        self.base_columns_details = data_source.column_details()
        self.view_columns_details = None

        # Special case for the base view
        if self.name == "base":
            self.view_table = self.base_table
            self._exists = True

        # We'll set up the view but it doesn't exist yet until a create method is called
        else:
            self.view_table = f"{self.base_table}_{self.name}"
            self._exists = False

    def exists(self):
        """Check if the view exists in the database
        Returns:
            bool: True if the view exists, False otherwise.
        """
        # Have we already checked if the view exists?
        if self._exists:
            return True

        # Query to check if the view exists
        check_view_query = f"""
        SELECT count(*) as view_count FROM information_schema.views
        WHERE table_schema = '{self.database}' AND table_name = '{self.view_table}'
        """

        # Execute the query
        result = self.data_source.query(check_view_query)

        # Check if the view count is greater than 0
        view_exists = result["view_count"][0] > 0
        self._exists = view_exists
        return view_exists

    def create_display(self, columns: list = None, column_limit: int = 30, recreate: bool = False):
        """Create a database view that manages which columns are used for display
        Args:
            columns(list): The columns to include in the view (default: None)
            column_limit(int): Max number of columns in the view (default: 30)
            recreate(bool): Drop and recreate the view (default: False)
        Returns:
            View: The View object for this view
        """

        # Check if the view already exists
        if self.exists() and not recreate:
            self.log.info(f"View {self.view_table} already exists")
            return

        # Create a new view
        self.log.important(f"Creating Display View {self.view_table}...")

        # If the user doesn't specify columns, then we'll limit the columns
        if columns is None:
            columns = self.data_source.column_names()[:column_limit]

        # Set the view columns details
        self.view_columns_details = {column: self.base_columns_details[column] for column in columns}

        # Enclose each column name in double quotes
        sql_columns = ", ".join([f'"{column}"' for column in columns])

        # Create the view query
        create_view_query = f"""
        CREATE OR REPLACE VIEW {self.view_table} AS
        SELECT {sql_columns} FROM {self.base_table}
        """

        # Execute the CREATE VIEW query
        self.data_source.execute_statement(create_view_query)

        # Update the _exists attribute
        self._exists = True

    def create_computation(self, columns: list = None, column_limit: int = 30, recreate: bool = False):
        """Create a database view that manages which columns are used for computation"""
        self.create_display(columns=columns, column_limit=column_limit, recreate=recreate)

    def create_training(self, id_column: str, columns: list = None, recreate: bool = False):
        """Create a database view that manages which columns are used
        Args:
            id_column(str): The name of the ID column
            columns(list): The columns to include in the view (default: None)
            recreate(bool): Drop and recreate the view (default: False)
        Returns:
            View: The View object for this view
        """

        # Check if the view already exists
        if self.exists() and not recreate:
            self.log.info(f"View {self.view_table} already exists")
            return

        # Create a new view
        self.log.important(f"Creating Training View {self.view_table}...")

        # If the user doesn't specify columns, then we'll use ALL the columns
        if columns is None:
            columns = self.data_source.column_names()

        # Set the view columns details
        self.view_columns_details = {column: self.base_columns_details[column] for column in columns}

        # Enclose each column name in double quotes
        sql_columns = ", ".join([f'"{column}"' for column in columns])

        # Logic to 'hash' the ID column from 1 to 10
        # We use this assign roughly 80% to training and 20% to hold-out
        hash_logic = f"from_big_endian_64(xxhash64(cast(cast({id_column} AS varchar) AS varbinary))) % 10"

        # Construct the CREATE VIEW query with consistent assignment based on hashed string ID
        create_view_query = f"""
        CREATE OR REPLACE VIEW {self.view_table} AS
        SELECT {sql_columns}, CASE
            WHEN {hash_logic} < 8 THEN 1  -- Assign roughly 80% to training
            ELSE 0  -- Assign roughly 20% to hold-out
        END AS training
        FROM {self.base_table}
        """

        # Execute the CREATE VIEW query
        self.data_source.execute_statement(create_view_query)

        # Update the _exists attribute
        self._exists = True

    def delete(self):
        """Delete the database view if it exists."""
        # Check if the view exists
        if not self.exists():
            self.log.info(f"View {self.view_table} does not exist, nothing to delete.")
            return

        # If the view exists, drop it
        self.log.important(f"Dropping View {self.view_table}...")
        drop_view_query = f"DROP VIEW {self.view_table}"

        # Execute the DROP VIEW query
        self.data_source.execute_statement(drop_view_query)

        # Update the _exists attribute
        self._exists = False

    def __repr__(self):
        """Return a string representation of this object"""
        return f"View: {self.database}:{self.view_table} (DataSource({self.data_source.uuid}) Exists: {self.exists()})"


class ViewManager:
    def __init__(self, data_source):
        """ViewManager: Used by DataSource to manage 'views' of the data
        Args:
            data_source(DataSource): The DataSource that this ViewManager is for
        """

        # Set up our instance attributes
        self.data_source = data_source
        self.database = data_source.get_database()
        self.base_table = data_source.get_table_name()
        self.display_view = None
        self.computation_view = None
        self.training_view = None

        # We have an internal 'base view' that's just the data source base table
        self.base_view = View("base", self.data_source)

    def create_display_view(self, recreate: bool = False):
        """Create the display view for this data source
        Args:
            recreate(bool): Drop and recreate the view (default: False)
        """
        self.display_view = View("display", self.data_source)
        self.display_view.create_display(recreate=recreate)

    def create_computation_view(self, recreate: bool = False):
        """Create the computation view for this data source
        Args:
            recreate(bool): Drop and recreate the view (default: False)
        Note: This just returns the display view for now
        """
        self.computation_view = View("computation", self.data_source)
        self.computation_view.create_computation(recreate=recreate)

    def create_training_view(self, id_column: str, recreate: bool = False):
        """Create the training view for this data source
        Args:
            id_column(str): The name of the ID column
            recreate(bool): Drop and recreate the view (default: False)
        """
        self.training_view = View("training", self.data_source)
        self.training_view.create_training(id_column, recreate=recreate)

    def get_view(self, view_type: ViewType) -> View:
        """Get the display view
        Args:
            view_type(ViewType): The type of view to return
        Returns:
            View: The display view for this data source
        """
        if view_type == ViewType.BASE:
            return self.data_source.get_table_name()
        return self.display_view

    def get_computation_view(self) -> View:
        """Get the computation view
        Returns:
            View: The computation view for this data source
        Note: This just returns the display view for now
        """
        return self.computation_view

    def get_training_view(self) -> View:
        """Get the training view
        Returns:
            View: The training view for this data source
        """
        return self.training_view

    def delete_all_views(self):
        """Delete all views for this data source"""
        if self.display_view:
            self.display_view.delete()
        if self.computation_view:
            self.computation_view.delete()
        if self.training_view:
            self.training_view.delete()


if __name__ == "__main__":
    """Exercise the ViewManager Class"""
    from pprint import pprint
    from sageworks.api.data_source import DataSource

    # Create a DataSource (which will create a ViewManager)
    data_source = DataSource("test_data", force_refresh=True)

    # Now create the default views
    data_source.view_manager.create_display_view()  # recreate=True for testing
    data_source.view_manager.create_computation_view()
    data_source.view_manager.create_training_view("id")

    # Get the display view
    my_view = data_source.get_display_view()
    print(my_view)

    # Get the computation view
    my_view = data_source.get_computation_view()
    print(my_view)

    # Get the training view
    my_view = data_source.get_training_view()
    print(my_view)

    # Get the column details for the training view
    pprint(my_view.view_columns_details)

    # Delete the training view
    my_view.delete()
