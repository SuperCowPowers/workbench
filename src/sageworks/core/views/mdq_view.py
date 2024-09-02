"""MDQView Class: A View that computes various data_quality metrics"""

from typing import Union
import pandas as pd

# SageWorks Imports
from sageworks.api import DataSource, Model
from sageworks.core.views.view import View
from sageworks.core.views.create_view import CreateView
from sageworks.core.views.view_utils import dataframe_to_table, get_column_list
from sageworks.algorithms.dataframe.row_tagger import RowTagger


class MDQView(CreateView):
    """MDQView Class: A View that computes various model data quality metrics"""

    def __init__(self):
        """Initialize the Model Data Quality View"""
        super().__init__()

    def get_view_name(self) -> str:
        """Get the name of the view"""
        return "mdq"

    def create_view_impl(self, data_source: DataSource, id_column: str, model: Model) -> Union[View, None]:
        """Create a Model Data Quality View: A View that computes various model data quality metrics

        Args:
            data_source (DataSource): The SageWorks DataSource object
            id_column (str): The name of the id column (must be defined for join logic)
            model (Model): The Model object to use for the target and features

        Returns:
            Union[View, None]: The created View object (or None if failed to create the view)
        """
        self.log.important(f"Creating Model Data Quality View {self.view_table_name}...")

        # Get the target and feature columns
        target = model.target()
        features = model.features()

        # Check the number of rows in the source_table, if greater than 1M, then give an error and return
        row_count = data_source.num_rows()
        if row_count > 1_000_000:
            self.log.error(
                f"Data Quality View cannot be created on more than 1M rows. {self.source_table} has {row_count} rows."
            )
            return None

        # Drop any columns generated from AWS
        aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
        source_table_columns = get_column_list(data_source, self.source_table)
        column_list = [col for col in source_table_columns if col not in aws_cols]

        # Enclose each column name in double quotes
        sql_columns = ", ".join([f'"{column}"' for column in column_list])

        # Pull in the data from the source_table
        query = f"SELECT {sql_columns} FROM {self.source_table}"
        df = data_source.query(query)

        # Check if the id_column exists in the source_table
        if id_column not in df.columns:
            self.log.error(f"id_column {id_column} not found in {self.source_table}. Cannot create Data Quality View.")
            return None

        # Check if the target column exists in the source_table
        if target not in df.columns:
            self.log.error(f"target column {target} not found in {self.source_table}. Cannot create Data Quality View.")
            return None

        # Check the type of the target column is categorical (not numeric)
        categorical_target = not pd.api.types.is_numeric_dtype(df[target])

        # Check if the feature columns exist in the source_table
        for feature in features:
            if feature not in df.columns:
                self.log.error(
                    f"feature column {feature} not found in {self.source_table}. Cannot create Data Quality View."
                )
                return None

        # Now run the RowTagger to compute coincident and high target gradient tags
        row_tagger = RowTagger(
            df,
            features=features,
            id_column=id_column,
            target_column=target,
            within_dist=0.25,
            min_target_diff=1.0,
            outlier_df=data_source.outliers(),
            categorical_target=categorical_target,
        )
        mdq_df = row_tagger.tag_rows()

        # HACK: These are the columns that are being added to the dataframe
        dq_columns = ["data_quality_tags", "data_quality"]
        mdq_df = mdq_df.drop(columns=dq_columns, errors="ignore")

        # We're going to rename the tags column to data_quality_tags
        mdq_df.rename(columns={"tags": "data_quality_tags"}, inplace=True)

        # We're going to compute a data_quality score based on the tags.
        # Specific/Domain specific logic can be added here.
        # If 'coincident' is in the tags, then the data_quality score is 0.0
        # If 'htg' is in the tags, then the data_quality score is 0.5
        # Else there's no bad tags so the data_quality score is 1.0
        mdq_df["data_quality"] = mdq_df["data_quality_tags"].apply(
            lambda tags: 0.0 if "coincident" in tags else 0.5 if "htg" in tags else 1.0
        )

        # Just want to keep the new data quality columns
        mdq_df = mdq_df[dq_columns + [id_column]]

        # Create the Model Data Quality supplemental table
        mdq_table = f"_{self.base_table}_{self.view_name}"
        dataframe_to_table(data_source, mdq_df, mdq_table)

        # Convert the list of dq_columns into a comma-separated string
        dq_columns_str = ", ".join([f"B.{col}" for col in dq_columns])

        # List the columns from A that are not in B to avoid overlap
        source_columns_str = ", ".join([f"A.{col}" for col in df.columns if col not in dq_columns])

        # Construct the CREATE VIEW query
        create_view_query = f"""
        CREATE OR REPLACE VIEW {self.view_table_name} AS
        SELECT {source_columns_str}, {dq_columns_str}
        FROM {self.source_table} A
        LEFT JOIN {mdq_table} B
        ON A.{id_column} = B.{id_column}
        """

        # Execute the CREATE VIEW query
        data_source.execute_statement(create_view_query)

        # Return the View
        return View(data_source, self.view_name, auto_create=False)


if __name__ == "__main__":
    """Exercise the Training View functionality"""
    from sageworks.api import FeatureSet, Model

    # Get the FeatureSet
    fs = FeatureSet("abalone_features")

    # Grab the Model
    model = Model("abalone-regression")

    # Create a MDQView
    mdq_view = MDQView().create_view(fs, id_column="id", model=model)

    # Pull the data quality dataframe
    my_df = mdq_view.pull_dataframe(head=True)
    print(my_df)

    # Delete the default data_quality view
    mdq_view.delete()
