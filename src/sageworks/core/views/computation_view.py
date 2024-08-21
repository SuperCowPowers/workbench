"""Create a Computation View: A View with a subset of columns for computation/stats purposes"""

from typing import Union

# SageWorks Imports
from sageworks.api import DataSource
from sageworks.core.views.view import View, ViewType
from sageworks.core.views.view_utils import get_column_list


class ComputationView(View):
    """Computation View Class: A View with a subset of columns for computation/stats purposes"""

    def __init__(self, data_source: DataSource):
        """Initialize the ComputationView

        Args:
            data_source (DataSource): The DataSource object
        """
        super().__init__(data_source, ViewType.COMPUTATION)

    def create_view(
        self,
        column_list: Union[list[str], None] = None,
        column_limit: int = 30,
        source_table: str = None,
    ):
        """Create a Computation View: A View with a subset of columns for computation/stats purposes

        Args:
            column_list (Union[list[str], None], optional): A list of columns to include. Defaults to None.
            column_limit (int, optional): The max number of columns to include. Defaults to 30.
            source_table (str, optional): The table/view to create the view from. Defaults to base table.
        """

        # Set the source_table to create the view from
        base_table = self.data_source.get_table_name()
        source_table = source_table if source_table else base_table

        # Create the computation view table name
        view_name = f"{base_table}_computation"

        self.log.important(f"Creating Computation View {view_name}...")

        # If the user doesn't specify columns, then we'll limit the columns
        if column_list is None:
            # Drop any columns generated from AWS
            aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
            source_table_columns = get_column_list(self.data_source, source_table)
            column_list = [col for col in source_table_columns if col not in aws_cols]

            # Limit the number of columns
            column_list = column_list[:column_limit]

        # Enclose each column name in double quotes
        sql_columns = ", ".join([f'"{column}"' for column in column_list])

        # Create the view query
        create_view_query = f"""
           CREATE OR REPLACE VIEW {view_name} AS
           SELECT {sql_columns} FROM {source_table}
           """

        # Execute the CREATE VIEW query
        self.data_source.execute_statement(create_view_query)


if __name__ == "__main__":
    """Exercise the Training View functionality"""
    from sageworks.api import FeatureSet

    # Get the FeatureSet
    fs = FeatureSet("test_features")

    # Create a Computation View
    comp_view = ComputationView(fs.data_source)
    print(comp_view)

    # Pull the display data
    df = comp_view.pull_dataframe()
    print(df.head())

    # Create a Display View with a subset of columns
    columns = ["id", "name", "age", "height", "weight"]
    comp_view.create_view(column_list=columns)
    print(comp_view.pull_dataframe(head=True))
