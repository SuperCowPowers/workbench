"""ViewManager: Used by DataSource to manage 'views' of the data
   Note: This is probably Athena specific, so we may need to refactor this
"""
import logging


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
        self.base_table = data_source.get_base_table_name()
        self.view_table = f"{self.base_table}_{self.name}"

        # A View object should be instantiated quickly, so
        # they don't really exist on creation.
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

    def create(self, columns: list = None, column_limit: int = None, recreate: bool = False):
        """Create a database view that manages which columns are used
        Args:
            columns(list): The columns to include in the view (default: None)
            column_limit(int): Max number of columns in the view (default: None)
            recreate(bool): Drop and recreate the view (default: False)
        Returns:
            View: The View object for this view
        """

        # Check if the view already exists
        if self.exists() and not recreate:
            self.log.info(f"View {self.view_table} already exists")
            return

        # Create a new view
        self.log.important(f"Creating View {self.view_table}...")

        # If the user doesn't specify columns, then we'll limit the columns
        if columns is None and column_limit is not None:
            columns = self.data_source.column_names()[:column_limit]

        # Create the view query
        create_view_query = f"""
        CREATE OR REPLACE VIEW {self.view_table} AS
        SELECT {', '.join(columns)} FROM {self.base_table}
        """

        # Execute the CREATE VIEW query
        self.data_source.execute_statement(create_view_query)

        # Update the _exists attribute
        self._exists = True

    def __repr__(self):
        """Return a string representation of this object"""
        return f"View(name={self.name}, database={self.database}, table={self.base_table})"


class ViewManager:
    def __init__(self, data_source):
        """ViewManager: Used by DataSource to manage 'views' of the data
        Args:
            data_source(DataSource): The DataSource that this ViewManager is for
        """

        # Set up our instance attributes
        self.data_source = data_source
        self.database = data_source.get_database()
        self.base_table = data_source.get_base_table_name()
        self.display_view = None
        self.training_view = None

    def create_default_views(self):
        """Create the default views for this data source"""

        # Create the display and training views
        self.display_view = View("display", self.data_source)
        self.display_view.create(column_limit=40)
        self.training_view = View("training", self.data_source)
        self.training_view.create()

    def get_display_view(self) -> View:
        """Get the display view
        Returns:
            View: The display view for this data source
        """
        return self.display_view

    def get_computation_view(self) -> View:
        """Get the computation view
        Returns:
            View: The computation view for this data source
        Note: This just returns the display view for now
        """
        return self.display_view

    def get_training_view(self) -> View:
        """Get the training view
        Returns:
            View: The training view for this data source
        """
        return self.training_view


if __name__ == "__main__":
    """Exercise the ViewManager Class"""
    from sageworks.artifacts.data_sources.athena_source import AthenaSource

    # Create a DataSource (which will create a ViewManager)
    data_source = AthenaSource("test_data")

    # Now create the default views
    data_source.view_manager.create_default_views()

    # Get the display view
    my_view = data_source.get_display_view()
    print(my_view)

    # Get the computation view
    my_view = data_source.get_computation_view()
    print(my_view)

    # Get the training view
    my_view = data_source.get_training_view()
    print(my_view)
