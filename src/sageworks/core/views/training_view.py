"""TrainingView Class: A View with an additional training columns that marks holdout ids"""

from typing import Union

# SageWorks Imports
from sageworks.api import DataSource
from sageworks.core.views.view import View, ViewType
from sageworks.core.views.view_utils import get_column_list


class TrainingView(View):
    """TrainingView Class: A View with an additional training columns that marks holdout ids"""

    def __init__(self, data_source: DataSource):
        """Initialize the TrainingView

        Args:
            data_source (DataSource): The DataSource object
        """
        super().__init__(data_source, ViewType.TRAINING)

    def create_view(self, id_column: str = None, holdout_ids: Union[list[str], None] = None, source_table: str = None):
        """Create a training view that marks hold out ids

        Args:
            id_column (str, optional): The name of the id column (default is None)
            holdout_ids (Union[list[str], None], optional): A list of holdout ids. Defaults to None.
            source_table (str, optional): The table/view to create the view from. Defaults to base table.
        """

        # Set the source_table to create the view from
        source_table = source_table if source_table else self.base_table

        # Create the training view table name
        view_name = f"{self.base_table}_training"
        self.log.important(f"Creating Training View {view_name}...")

        # Drop any columns generated from AWS
        aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
        source_table_columns = get_column_list(self.data_source, source_table)
        column_list = [col for col in source_table_columns if col not in aws_cols]

        # Sanity check on the id column
        if not id_column:
            self.log.important("No id column specified, we'll try the auto_id_column ..")
            if not self.auto_id_column:
                self.log.error("No id column specified and no auto_id_column found, aborting ..")
                return
            else:
                if self.auto_id_column not in column_list:
                    self.log.error(f"Auto id column {self.auto_id_column} not found in column list, aborting ..")
                    return
                else:
                    id_column = self.auto_id_column

        # If we don't have holdout ids, create a default training view
        if not holdout_ids:
            self._default_training_view(id_column, view_name)
            return

        # Format the list of hold out ids for SQL IN clause
        if holdout_ids and all(isinstance(id, str) for id in holdout_ids):
            formatted_holdout_ids = ", ".join(f"'{id}'" for id in holdout_ids)
        else:
            formatted_holdout_ids = ", ".join(map(str, holdout_ids))

        # Enclose each column name in double quotes
        sql_columns = ", ".join([f'"{column}"' for column in column_list])

        # Construct the CREATE VIEW query
        create_view_query = f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT {sql_columns}, CASE
            WHEN {id_column} IN ({formatted_holdout_ids}) THEN 0
            ELSE 1
        END AS training
        FROM {source_table}
        """

        # Execute the CREATE VIEW query
        self.data_source.execute_statement(create_view_query)

    # This is an internal method that's used to create a default training view
    def _default_training_view(self, id_column: str, view_name: str):
        """Create a default view in Athena that assigns roughly 80% of the data to training

        Args:
            id_column (str): The name of the id column
            view_name (str): The name of the view to create
        """
        self.log.important(f"Creating default Training View {view_name}...")

        # Drop any columns generated from AWS
        aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
        column_list = [col for col in self.data_source.column_names() if col not in aws_cols]

        # Enclose each column name in double quotes
        sql_columns = ", ".join([f'"{column}"' for column in column_list])

        #    Construct the CREATE VIEW query with a simple modulo operation for the 80/20 split
        #    using the id column as the stable identifier for row numbering
        create_view_query = f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT {sql_columns}, CASE
            WHEN MOD(ROW_NUMBER() OVER (ORDER BY {id_column}), 10) < 8 THEN 1  -- Assign 80% to training
            ELSE 0  -- Assign roughly 20% to validation/test
        END AS training
        FROM {self.base_table}
        """

        # Execute the CREATE VIEW query
        self.data_source.execute_statement(create_view_query)


if __name__ == "__main__":
    """Exercise the Training View functionality"""
    from sageworks.api import FeatureSet

    # Get the FeatureSet
    fs = FeatureSet("test_features")

    # Delete the existing training view
    training_view = TrainingView(fs)
    training_view.delete()

    # Create a TrainingView
    training_view = TrainingView(fs)
    print(training_view)

    # Pull the training data
    df = training_view.pull_dataframe()
    print(df.head())

    # Create a TrainingView with holdout ids
    holdout_ids = list(range(10))
    training_view.create_view(id_column="id", holdout_ids=holdout_ids)

    # Pull the training data
    df = training_view.pull_dataframe()
    print(df.head())
