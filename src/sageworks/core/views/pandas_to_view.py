"""PandasToView Class: A View that joins the source_table with a Pandas dataframe"""

from typing import Union
import pandas as pd

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet
from sageworks.core.views.view import View
from sageworks.core.views.create_view import CreateView
from sageworks.core.views.view_utils import dataframe_to_table, get_column_list


class PandasToView(CreateView):
    """PandasToView Class: A View that joins the source_table with a Pandas dataframe

    Common Usage:
    ```
    # Grab a DataSource
    ds = DataSource("test_data")

    # Do some awesome Feature Engineering :)
    my_df = ds.pull_dataframe()
    my_df["random1"] = np.random.rand(len(my_df))
    my_df["random2"] = np.random.rand(len(my_df))

    # Create your new View
    fe_view = PandasToView.create("feature_engineering_view", ds, df=my_df, id_column="id")

    # Query the view
    df = fe_view.query(f"SELECT * FROM {fe_view.table} where residuals > 0.5")
    ```
    """

    @classmethod
    def create(
        cls,
        view_name: str,
        artifact: Union[DataSource, FeatureSet],
        df: pd.DataFrame,
        id_column: str,
        source_table: str = None,
    ) -> Union[View, None]:
        """Factory method to create and return a PandasToView instance.

        Args:
            view_name (str): The name of the view
            artifact (Union[DataSource, FeatureSet]): The DataSource or FeatureSet object
            df (pd.DataFrame): The Pandas DataFrame to join with the source_table
            id_column (str): The name of the id column (must be defined for join logic)
            source_table (str, optional): The table/view to create the view from. Defaults to None

        Returns:
            Union[View, None]: The created View object (or None if failed to create the view)
        """
        # Instantiate the PandasToView
        instance = cls(view_name, artifact, source_table)

        # Delegate to the internal method for creating the view
        return instance._create_view(df, id_column)

    def _create_view(self, df: pd.DataFrame, id_column: str) -> Union[View, None]:
        """Internal method to create the view by joining with a Pandas DataFrame.

        Args:
            df (pd.DataFrame): The Pandas DataFrame to join with the source_table
            id_column (str): The name of the id column (must be defined for join logic)

        Returns:
            Union[View, None]: The created View object (or None if failed to create the view)
        """
        self.log.important(f"Creating View with DF {self.table}...")

        # Check the number of rows in the source_table, if greater than 1M, then give an error and return
        row_count = self.data_source.num_rows()
        if row_count > 1_000_000:
            self.log.error(
                f"View with DF cannot be created on more than 1M rows. {self.source_table} has {row_count} rows."
            )
            return None

        # Drop any columns generated from AWS
        aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
        source_table_columns = get_column_list(self.data_source, self.source_table)
        column_list = [col for col in source_table_columns if col not in aws_cols]

        # Drop any columns generated from AWS from the incoming dataframe
        df = df.drop(columns=aws_cols, errors="ignore")

        # Enclose each column name in double quotes
        sql_columns = ", ".join([f'"{column}"' for column in column_list])

        # Pull in the data from the source_table
        query = f"SELECT {sql_columns} FROM {self.source_table}"
        source_df = self.data_source.query(query)

        # Check if the id_column exists in the source_table
        if id_column not in source_df.columns:
            self.log.error(f"id_column {id_column} not found in {self.source_table}. Cannot create the View.")
            return None

        # Check if the id_column exists in the dataframe
        if id_column not in df.columns:
            self.log.error(f"id_column {id_column} not found in the dataframe. Cannot create the View.")
            return None

        # Remove any columns in the incoming df that overlap with the source_df (except for the id_column)
        overlap_columns = [col for col in df.columns if col in source_df.columns and col != id_column]
        self.log.info(f"Removing overlap columns: {overlap_columns}")
        df = df.drop(columns=overlap_columns)

        # Create a supplemental data table with the incoming dataframe
        df_table = f"_{self.base_table_name}_{self.view_name}"
        dataframe_to_table(self.data_source, df, df_table)

        # Create a list of columns in SQL form (for the source table)
        source_columns_str = ", ".join([f'A."{col}"' for col in source_df.columns])

        # Create a list of columns in SQL form (for the incoming dataframe table)
        df_columns_str = ", ".join([f'B."{col}"' for col in df.columns if col != id_column])

        # Construct the CREATE VIEW query
        create_view_query = f"""
        CREATE OR REPLACE VIEW {self.table} AS
        SELECT {source_columns_str}, {df_columns_str}
        FROM {self.source_table} A
        LEFT JOIN {df_table} B
        ON A.{id_column} = B.{id_column}
        """

        # Execute the CREATE VIEW query
        self.data_source.execute_statement(create_view_query)

        # Return the View
        return View(self.data_source, self.view_name, auto_create_view=False)


if __name__ == "__main__":
    """Exercise the PandasToView functionality"""
    from sageworks.api import DataSource
    import numpy as np

    # Grab a DataSource
    ds = DataSource("test_data")

    # Generate a DataFrame with two random columns
    my_df = ds.pull_dataframe()
    my_df["random1"] = np.random.rand(len(my_df))
    my_df["random2"] = np.random.rand(len(my_df))

    # Create a PandasToView
    df_view = PandasToView.create("test_df", ds, df=my_df, id_column="id")

    # Pull the dataframe view
    my_df = df_view.pull_dataframe(head=True)
    print(my_df)

    # Delete the dataframe view
    df_view.delete()
