"""View: Used by DataSource to manage 'views' of the data"""


class View:
    def __init__(self, name: str, database: str, table: str):
        """View: A Helper Class for the ViewManager
        Args:
            name(str): The name of the view
            database(str): The name of the database
            table(str): The name of the table
        """

        # Set up our instance attributes
        self.name = name
        self.database = database
        self.table = table

    def __repr__(self):
        """Return a string representation of this object"""
        return f"View(name={self.name}, database={self.database}, table={self.table})"


class ViewManager:
    def __init__(self, base_view: View, display_view: View, training_view: View):
        """ViewManager: Used by DataSource to manage 'views' of the data"""

        # Set up our instance attributes
        self.base_view = base_view
        self.display_view = display_view
        self.training_view = training_view

    def get_base_view(self) -> View:
        """Get the base view
        Returns:
            View: The base/default view for this data source
        """
        return self.base_view

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
    """Exercise the AthenaSource Class"""
    from sageworks.artifacts.data_sources.athena_source import AthenaSource

    # Set up our ViewManager
    database = 'fake_db'
    data_uuid = 'fake_data_uuid'
    view_manager = ViewManager(
        base_view=View(name="base", database=database, table=data_uuid),
        display_view=View(name="display", database=database, table=f"{data_uuid}_display"),
        training_view=View(name="training", database=database, table=f"{data_uuid}_training"),
    )

    # Get the base/default view
    my_view = view_manager.get_base_view()
    print(my_view)

    # Get the display view
    my_view = view_manager.get_display_view()
    print(my_view)

    # Get the computation view
    my_view = view_manager.get_computation_view()
    print(my_view)

    # Get the training view
    my_view = view_manager.get_training_view()
    print(my_view)