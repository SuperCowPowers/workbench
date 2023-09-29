"""This Script creates the Classification Artifacts in AWS

FeatureSets:
    - abalone_classification
Models:
    - abalone-classification
Endpoints:
    - abalone-classification-end
"""
import sys
import time
import pandas as pd

from pathlib import Path
from sageworks.artifacts.feature_sets.feature_set import FeatureSet
from sageworks.artifacts.models.model import Model
from sageworks.artifacts.endpoints.endpoint import Endpoint

from sageworks.transforms.pandas_transforms.data_to_pandas import DataToPandas
from sageworks.transforms.pandas_transforms.pandas_to_features import PandasToFeatures
from sageworks.transforms.features_to_model.features_to_model import FeaturesToModel
from sageworks.transforms.model_to_endpoint.model_to_endpoint import ModelToEndpoint

if __name__ == "__main__":
    # Get the path to the dataset in the repository data directory
    abalone_data_path = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "abalone.csv"

    # Create the abalone_classification FeatureSet
    if not FeatureSet("abalone_classification").exists():
        # Grab the data from the DataSource
        data_to_pandas = DataToPandas("abalone_data")
        data_to_pandas.transform()
        df = data_to_pandas.get_output()

        # Convert the regression target to a categorical type and drop the old column
        bins = [0, 7, 12, float("inf")]
        labels = ["young", "adult", "old"]
        df["clam_age_class"] = pd.cut(df["class_number_of_rings"], bins=bins, labels=labels)
        df.drop("class_number_of_rings", axis=1, inplace=True)

        # Create the FeatureSet
        pandas_to_features = PandasToFeatures("abalone_classification")
        pandas_to_features.set_input(df, target_column="clam_age_class")
        pandas_to_features.set_output_tags(["abalone", "classification"])
        pandas_to_features.transform()

    # Create the abalone_classification Model
    if not Model("abalone-classification").exists():
        features_to_model = FeaturesToModel("abalone_classification", "abalone-classification")
        features_to_model.set_output_tags(["abalone", "classification"])
        features_to_model.transform(
            target="clam_age_class", description="Abalone Classification Model", model_type="classifier"
        )
        print("Waiting for the Model to be created...")
        time.sleep(10)

    # Create the abalone_regression Endpoint
    if not Endpoint("abalone-classification-end").exists():
        model_to_endpoint = ModelToEndpoint("abalone-classification", "abalone-classification-end")
        model_to_endpoint.set_output_tags(["abalone", "classification"])
        model_to_endpoint.transform()
