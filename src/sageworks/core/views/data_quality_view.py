"""DataQualityView Class: A View that computes various data_quality metrics"""

# SageWorks Imports
from sageworks.api import DataSource
from sageworks.core.views.view import View, ViewType
from sageworks.core.views.view_utils import dataframe_to_table, get_column_list
from sageworks.algorithms.dataframe.row_tagger import RowTagger


class DataQualityView(View):
    """DataQualityView Class: A View that computes various data_quality metrics"""

    def __init__(self, data_source: DataSource):
        """Initialize the DataQualityView

        Args:
            data_source (DataSource): The DataSource object
        """
        super().__init__(data_source, ViewType.DATA_QUALITY)

    def create_view(self, id_column: str, target: str, features: list, source_table: str = None):
        """Create a Data Quality View: A View that computes various data_quality metrics

        Args:
            id_column (str): The name of the id column (must be defined for join logic)
            target (str): The name of the target column
            features (list): The list of feature columns
            source_table_name (str, optional): The table/view to create the view from. Defaults to base table.
        """

        # Get the source_table to create the view from
        base_table = self.data_source.get_table_name()
        source_table = source_table if source_table else base_table

        # Check the number of rows in the source_table, if greater than 1M, then give an error and return
        row_count = self.data_source.num_rows()
        if row_count > 1_000_000:
            self.log.error(
                f"Data Quality View cannot be created on more than 1M rows. {source_table} has {row_count} rows."
            )
            return

        # Drop any columns generated from AWS
        aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
        source_table_columns = get_column_list(self.data_source, source_table)
        column_list = [col for col in source_table_columns if col not in aws_cols]

        # Enclose each column name in double quotes
        sql_columns = ", ".join([f'"{column}"' for column in column_list])

        # Pull in the data from the source_table
        query = f"SELECT {sql_columns} FROM {source_table}"
        df = self.data_source.query(query)

        # Check if the id_column exists in the source_table
        if id_column not in df.columns:
            self.log.error(f"id_column {id_column} not found in {source_table}. Cannot create Data Quality View.")
            return

        # Check if the target column exists in the source_table
        if target not in df.columns:
            self.log.error(f"target column {target} not found in {source_table}. Cannot create Data Quality View.")
            return

        # Check if the feature columns exist in the source_table
        for feature in features:
            if feature not in df.columns:
                self.log.error(
                    f"feature column {feature} not found in {source_table}. Cannot create Data Quality View."
                )
                return

        # Now run the RowTagger to compute coincident and high target gradient tags
        row_tagger = RowTagger(
            df,
            features=feature_columns,
            id_column=id_column,
            target_column=target_column,
            within_dist=0.25,
            min_target_diff=1.0,
        )
        dq_df = row_tagger.tag_rows()

        # We're going to rename the tags column to data_quality_tags
        dq_df.rename(columns={"tags": "data_quality_tags"}, inplace=True)

        # We're going to compute a data_quality score based on the tags. The tags are a set() of strings
        # If 'coincident' is in the tags, then the data_quality score is 0.0
        # If 'htg' is in the tags, then the data_quality score is 0.5
        # Else there's no bad tags so the data_quality score is 1.0
        dq_df["data_quality"] = dq_df["data_quality_tags"].apply(
            lambda tags: 0.0 if "coincident" in tags else 0.5 if "htg" in tags else 1.0
        )

        # Create the data_quality supplemental table
        data_quality_table = f"_{base_table}_data_quality"
        dataframe_to_table(self.data_source, dq_df[[id_column, "data_quality"]], data_quality_table)

        # Create the data_quality view (join the data_quality table with the source_table)
        view_name = f"{base_table}_data_quality"
        self.log.important(f"Creating Data Quality View {view_name}...")
        create_view_query = f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT A.*, B.data_quality
        FROM {source_table} A
        LEFT JOIN {data_quality_table} B
        ON A.{id_column} = B.{id_column}
        """

        # Execute the CREATE VIEW query
        self.data_source.execute_statement(create_view_query)


if __name__ == "__main__":
    """Exercise the Training View functionality"""
    from sageworks.api import FeatureSet, Model

    # Get the FeatureSet
    fs = FeatureSet("aqsol_mol_descriptors")

    # Get the target and feature columns
    m = Model("aqsol-mol-regression")
    target_column = m.target()
    feature_columns = m.features()

    # Create a DataQualityView
    dq_view = DataQualityView(fs)

    # Pull the data quality dataframe (not sure what this will do)
    df = dq_view.pull_dataframe(head=True)
    print(df)

    # Create a default data_quality view
    dq_view.create_view("id", target_column, feature_columns)
    df = dq_view.pull_dataframe(head=True)
    print(df)
