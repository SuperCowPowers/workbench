"""MDQView Class: A View that computes various model data quality metrics"""
from typing import Union
import pandas as pd

# SageWorks Imports
from sageworks.api import FeatureSet, Model
from sageworks.core.views.view import View
from sageworks.core.views.pandas_to_view import PandasToView
from sageworks.algorithms.dataframe.row_tagger import RowTagger
from sageworks.algorithms.dataframe.residuals_calculator import ResidualsCalculator


class MDQView:
    """MDQView Class: A View that computes various model data quality metrics

    Common Usage:
        ```
        # Grab a FeatureSet and a Model
        fs = FeatureSet("abalone_features")
        model = Model("abalone-regression")

        # Create a ModelDataQuality View
        mdq_view = MDQView.create(fs, model=model, id_column="id")
        my_df = mdq_view.pull_dataframe(head=True)

        # Query the view
        df = mdq_view.query(f"SELECT * FROM {mdq_view.table} where residuals > 0.5")
        ```
    """

    @classmethod
    def create(
        cls,
        fs: FeatureSet,
        model: Model,
        id_column: str,
    ) -> Union[View, None]:
        """Create a Model Data Quality View with metrics

        Args:
            fs (FeatureSet): The FeatureSet object
            model (Model): The Model object to use for the target and features
            id_column (str): The name of the id column (must be defined for join logic)

        Returns:
            Union[View, None]: The created View object (or None if failed)
        """
        # Log view creation
        fs.log.important("Creating Model Data Quality View...")

        # Get the target and feature columns from the model
        target = model.target()
        features = model.features()

        # Pull in data from the source table
        df = fs.data_source.query(f"SELECT * FROM {fs.data_source.uuid}")

        # Super Hack
        if "udm_asy_res_value" in df.columns:
            import numpy as np
            df["udm_asy_res_value"] = df["udm_asy_res_value"].replace(0, 1e-10)
            df["log_s"] = np.log10(df["udm_asy_res_value"] / 1e6)
            target = "log_s"

        # Check if the target and features are available in the data source
        missing_columns = [col for col in [target] + features if col not in df.columns]
        if missing_columns:
            fs.log.error(f"Missing columns in data source: {missing_columns}")
            return None

        # Check if the target is categorical
        categorical_target = not pd.api.types.is_numeric_dtype(df[target])

        # Compute row tags with RowTagger
        row_tagger = RowTagger(
            df,
            features=features,
            id_column=id_column,
            target_column=target,
            within_dist=0.25,
            min_target_diff=1.0,
            outlier_df=fs.data_source.outliers(),
            categorical_target=categorical_target,
        )
        mdq_df = row_tagger.tag_rows()

        # Rename and compute data quality scores based on tags
        mdq_df.rename(columns={"tags": "data_quality_tags"}, inplace=True)

        # We're going to compute a data_quality score based on the tags.
        mdq_df["data_quality"] = mdq_df["data_quality_tags"].apply(cls.calculate_data_quality)

        # Compute residuals using ResidualsCalculator
        residuals_calculator = ResidualsCalculator(n_splits=5, random_state=42)
        residuals_df = residuals_calculator.fit_transform(df[features], df[target])

        # Add id_column to the residuals dataframe and merge with mdq_df
        residuals_df[id_column] = df[id_column]

        # Drop overlapping columns in mdq_df (except for the id_column) to avoid _x and _y suffixes
        overlap_columns = [col for col in residuals_df.columns if col in mdq_df.columns and col != id_column]
        mdq_df = mdq_df.drop(columns=overlap_columns)

        # Merge the DataFrames, with the id_column as the join key
        mdq_df = mdq_df.merge(residuals_df, on=id_column, how="left")

        # Delegate view creation to PandasToView
        return PandasToView.create("mdq_view", fs, df=mdq_df, id_column=id_column)

    @staticmethod
    def calculate_data_quality(tags):
        score = 1.0  # Start with the default score
        if "coincident" in tags:
            score -= 1.0
        if "htg" in tags:
            score -= 0.5
        if "outlier" in tags:
            score -= 0.25
        return score


if __name__ == "__main__":
    """Exercise the MDQView functionality"""
    from sageworks.api import FeatureSet, Model

    # Get the FeatureSet
    my_fs = FeatureSet("abalone_features")

    # Grab the Model
    my_model = Model("abalone-regression")

    # Create a MDQView
    mdq_view = MDQView.create(my_fs, model=my_model, id_column="id")

    # Pull the data quality dataframe
    my_df = mdq_view.pull_dataframe(head=True)
    print(my_df)
