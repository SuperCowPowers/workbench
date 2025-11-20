"""TrainingView Class: A View with an additional training column that marks holdout ids"""

from typing import Union

# Workbench Imports
from workbench.api import FeatureSet
from workbench.core.views.view import View
from workbench.core.views.create_view import CreateView
from workbench.core.views.view_utils import get_column_list


class TrainingView(CreateView):
    """TrainingView Class: A View with an additional training column (80/20 or holdout ids).
    The TrainingView class creates a SQL view that includes all columns from the source table
    along with an additional boolean column named "training". This view can also include
    a SQL filter expression to filter the rows included in the view.


    Common Usage:
        ```python
        # Create a default TrainingView
        fs = FeatureSet("test_features")
        training_view = TrainingView.create(fs)
        df = training_view.pull_dataframe()

        # Create a TrainingView with a specific filter expression
        training_view = TrainingView.create(fs, id_column="auto_id", filter_expression="age > 30")
        df = training_view.pull_dataframe()

        # Query the view
        df = training_view.query(f"SELECT * FROM {training_view.table} where training = TRUE")
        ```
    """

    @classmethod
    def create(
        cls,
        feature_set: FeatureSet,
        *,  # Enforce keyword arguments after feature_set
        id_column: str = None,
        holdout_ids: Union[list[str], list[int], None] = None,
        filter_expression: str = None,
        source_table: str = None,
    ) -> Union[View, None]:
        """Factory method to create and return a TrainingView instance.

        Args:
            feature_set (FeatureSet): A FeatureSet object
            id_column (str, optional): The name of the id column. Defaults to None.
            holdout_ids (Union[list[str], list[int], None], optional): A list of holdout ids. Defaults to None.
            filter_expression (str, optional): SQL filter expression (e.g., "age > 25 AND status = 'active'").
                                               Defaults to None.
            source_table (str, optional): The table/view to create the view from. Defaults to None.

        Returns:
            Union[View, None]: The created View object (or None if failed to create the view)
        """
        # Instantiate the TrainingView with "training" as the view name
        instance = cls("training", feature_set, source_table)

        # Drop any columns generated from AWS
        aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
        source_table_columns = get_column_list(instance.data_source, instance.source_table)
        column_list = [col for col in source_table_columns if col not in aws_cols]

        # Sanity check on the id column
        if not id_column:
            instance.log.important("No id column specified, we'll try the auto_id_column ..")
            if not instance.auto_id_column:
                instance.log.error("No id column specified and no auto_id_column found, aborting ..")
                return None
            else:
                if instance.auto_id_column not in column_list:
                    instance.log.error(
                        f"Auto id column {instance.auto_id_column} not found in column list, aborting .."
                    )
                    return None
                else:
                    id_column = instance.auto_id_column

        # Enclose each column name in double quotes
        sql_columns = ", ".join([f'"{column}"' for column in column_list])

        # Build the training assignment logic
        if holdout_ids:
            # Format the list of holdout ids for SQL IN clause
            if all(isinstance(id, str) for id in holdout_ids):
                formatted_holdout_ids = ", ".join(f"'{id}'" for id in holdout_ids)
            else:
                formatted_holdout_ids = ", ".join(map(str, holdout_ids))

            training_logic = f"""CASE
                WHEN {id_column} IN ({formatted_holdout_ids}) THEN False
                ELSE True
            END AS training"""
        else:
            # Default 80/20 split using modulo
            training_logic = f"""CASE
                WHEN MOD(ROW_NUMBER() OVER (ORDER BY {id_column}), 10) < 8 THEN True
                ELSE False
            END AS training"""

        # Build WHERE clause if filter_expression is provided
        where_clause = f"\nWHERE {filter_expression}" if filter_expression else ""

        # Construct the CREATE VIEW query
        create_view_query = f"""
        CREATE OR REPLACE VIEW {instance.table} AS
        SELECT {sql_columns}, {training_logic}
        FROM {instance.source_table}{where_clause}
        """

        # Execute the CREATE VIEW query
        instance.data_source.execute_statement(create_view_query)

        # Return the View
        return View(instance.data_source, instance.view_name, auto_create_view=False)

    @classmethod
    def create_with_sql(
        cls,
        feature_set: FeatureSet,
        *,
        sql_query: str,
        id_column: str = None,
    ) -> Union[View, None]:
        """Factory method to create a TrainingView from a custom SQL query.

        This method takes a complete SQL query and adds the default 80/20 training split.
        Use this when you need complex queries like UNION ALL for oversampling.

        Args:
            feature_set (FeatureSet): A FeatureSet object
            sql_query (str): Complete SELECT query (without the final semicolon)
            id_column (str, optional): The name of the id column for training split. Defaults to None.

        Returns:
            Union[View, None]: The created View object (or None if failed)
        """
        # Instantiate the TrainingView
        instance = cls("training", feature_set)

        # Sanity check on the id column
        if not id_column:
            instance.log.important("No id column specified, using auto_id_column")
            if not instance.auto_id_column:
                instance.log.error("No id column specified and no auto_id_column found, aborting")
                return None
            id_column = instance.auto_id_column

        # Default 80/20 split using modulo
        training_logic = f"""CASE
            WHEN MOD(ROW_NUMBER() OVER (ORDER BY {id_column}), 10) < 8 THEN True
            ELSE False
        END AS training"""

        # Wrap the custom query and add training column
        create_view_query = f"""
        CREATE OR REPLACE VIEW {instance.table} AS
        SELECT *, {training_logic}
        FROM ({sql_query}) AS custom_source
        """

        # Execute the CREATE VIEW query
        instance.data_source.execute_statement(create_view_query)

        # Return the View
        return View(instance.data_source, instance.view_name, auto_create_view=False)


if __name__ == "__main__":
    """Exercise the Training View functionality"""
    from workbench.api import FeatureSet

    # Get the FeatureSet
    fs = FeatureSet("abalone_features")

    # Delete the existing training view
    training_view = TrainingView.create(fs)
    training_view.delete()

    # Create a default TrainingView
    training_view = TrainingView.create(fs)
    print(training_view)

    # Pull the training data
    df = training_view.pull_dataframe()
    print(df.head())
    print(df["training"].value_counts())

    # Create a TrainingView with holdout ids
    my_holdout_ids = list(range(10))
    training_view = TrainingView.create(fs, id_column="auto_id", holdout_ids=my_holdout_ids)

    # Pull the training data
    df = training_view.pull_dataframe()
    print(df.head())
    print(df["training"].value_counts())
    print(f"Shape: {df.shape}")
    print(f"Diameter min: {df['diameter'].min()}, max: {df['diameter'].max()}")

    # Test the filter expression
    training_view = TrainingView.create(fs, id_column="auto_id", filter_expression="diameter > 0.5")
    df = training_view.pull_dataframe()
    print(df.head())
    print(f"Shape with filter: {df.shape}")
    print(f"Diameter min: {df['diameter'].min()}, max: {df['diameter'].max()}")

    # Test create_with_sql with a custom query (UNION ALL for oversampling)
    print("\n--- Testing create_with_sql with oversampling ---")
    base_table = fs.table
    replicate_ids = [0, 1, 2]  # Oversample these IDs

    custom_sql = f"""
        SELECT * FROM {base_table}

        UNION ALL

        SELECT * FROM {base_table}
        WHERE auto_id IN ({', '.join(map(str, replicate_ids))})
    """

    training_view = TrainingView.create_with_sql(fs, sql_query=custom_sql, id_column="auto_id")
    df = training_view.pull_dataframe()
    print(f"Shape with custom SQL: {df.shape}")
    print(df["training"].value_counts())

    # Verify oversampling - check if replicated IDs appear twice
    for rep_id in replicate_ids:
        count = len(df[df["auto_id"] == rep_id])
        print(f"ID {rep_id} appears {count} times")
