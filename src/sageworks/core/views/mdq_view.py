"""MDQView Class: A View that computes various data_quality metrics"""

from typing import Union
import pandas as pd

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet, Model
from sageworks.core.views.view import View
from sageworks.core.views.create_view_with_df import CreateViewWithDF
from sageworks.algorithms.dataframe.row_tagger import RowTagger
from sageworks.algorithms.dataframe.residuals_calculator import ResidualsCalculator


class MDQView:
    """MDQView Class: A View that computes various model data quality metrics"""

    def __init__(self, artifact: Union[DataSource, FeatureSet], source_table: str = None):
        """Initialize the ColumnSubsetView

        Args:
            artifact (Union[DataSource, FeatureSet]): The DataSource or FeatureSet object
            source_table (str, optional): The table/view to create the view from. Defaults to None
        """
        self.log = artifact.log

        # We're going to use the CreateViewWithDF class internally
        self.cv_with_df = CreateViewWithDF("mdq", artifact, source_table)
        self.data_source = self.cv_with_df.data_source
        self.source_table = self.cv_with_df.source_table

    def create(self, model: Model, id_column: str) -> Union[View, None]:
        """Create a Model Data Quality View: A View that computes various model data quality metrics

        Args:
            model (Model): The Model object to use for the target and features
            id_column (str): The name of the id column (must be defined for join logic)

        Returns:
            Union[View, None]: The created View object (or None if failed to create the view)
        """
        self.log.important("Creating Model Data Quality View...")

        # Get the target and feature columns
        target = model.target()
        features = model.features()

        # Make sure the target and features are in the data_source
        df = self.data_source.query(f"SELECT * FROM {self.source_table}")
        ds_columns = df.columns
        if target not in ds_columns:
            self.log.error(f"Target column {target} not found in {self.data_source.uuid}. Cannot create MDQ View.")
            return None
        for feature in features:
            if feature not in ds_columns:
                self.log.error(
                    f"Feature column {feature} not found in {self.data_source.uuid}. Cannot create MDQ View."
                )
                return None

        # Check the type of the target column is categorical (not numeric)
        categorical_target = not pd.api.types.is_numeric_dtype(df[target])

        # Now run the RowTagger to compute coincident and high target gradient tags
        row_tagger = RowTagger(
            df,
            features=features,
            id_column=id_column,
            target_column=target,
            within_dist=0.25,
            min_target_diff=1.0,
            outlier_df=self.data_source.outliers(),
            categorical_target=categorical_target,
        )
        mdq_df = row_tagger.tag_rows()

        # Just some renaming
        mdq_df.rename(columns={"tags": "data_quality_tags"}, inplace=True)

        # We're going to compute a data_quality score based on the tags.
        # Specific/Domain specific logic can be added here.
        # If 'coincident' is in the tags, then the data_quality score is 0.0
        # If 'htg' is in the tags, then the data_quality score is 0.5
        # Else there's no bad tags so the data_quality score is 1.0
        mdq_df["data_quality"] = mdq_df["data_quality_tags"].apply(
            lambda tags: 0.0 if "coincident" in tags else 0.5 if "htg" in tags else 1.0
        )

        # Spin up the ResidualsCalculator
        residuals_calculator = ResidualsCalculator(n_splits=5, random_state=42)
        residuals_df = residuals_calculator.fit_transform(df[features], df[target])

        # Add the id_column to the residuals_df
        residuals_df[id_column] = df[id_column]

        # Get the list of columns to add from residuals_df, excluding any columns already in mdq_df
        new_columns = [id_column] + [
            col for col in residuals_df.columns if col != id_column and col not in mdq_df.columns
        ]

        # Merge the DataFrames, only including new columns from residuals_df
        mdq_df = mdq_df.merge(residuals_df[new_columns], on=id_column, how="left")

        # Call our internal CreateViewWithDF to create the Model Data Quality View
        return self.cv_with_df.create(df=mdq_df, id_column=id_column)


if __name__ == "__main__":
    """Exercise the Training View functionality"""
    from sageworks.api import FeatureSet, Model

    # Get the FeatureSet
    fs = FeatureSet("abalone_features")

    # Grab the Model
    model = Model("abalone-regression")

    # Create a MDQView
    mdq_view = MDQView(fs).create(model=model, id_column="id")

    # Pull the data quality dataframe
    my_df = mdq_view.pull_dataframe(head=True)
    print(my_df)

    # Delete the default data_quality view
    # mdq_view.delete()
