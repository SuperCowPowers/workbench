"""This Script DeleteS the SageWorks Artifacts in AWS needed for the tests

DataSources:
    - test_data
    - abalone_data
FeatureSets:
    - test_feature_set
    - abalone_feature_set
Models:
    - abalone-regression
Endpoints:
    - abalone-regression-end
"""
import sys
import time
from pathlib import Path
from sageworks.artifacts.data_sources.data_source import DataSource
from sageworks.artifacts.feature_sets.feature_set import FeatureSet
from sageworks.artifacts.models.model import Model
from sageworks.artifacts.endpoints.endpoint import Endpoint


if __name__ == "__main__":
    # Get the path to the dataset in the repository data directory
    test_data_path = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "test_data.csv"
    abalone_data_path = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "abalone.csv"

    # Delete the test_data DataSource
    ds = DataSource("test_data")
    if ds.exists():
        print("Deleting test_data...")
        ds.delete()

    # Delete the abalone_data DataSource
    ds = DataSource("abalone_data")
    if ds.exists():
        print("Deleting abalone_data...")
        ds.delete()

    # Delete the test_feature_set FeatureSet
    fs = FeatureSet("test_feature_set")
    if fs.exists():
        print("Deleting test_feature_set...")
        fs.delete()

    # Delete the abalone_feature_set FeatureSet
    fs = FeatureSet("abalone_feature_set")
    if fs.exists():
        print("Deleting abalone_feature_set...")
        fs.delete()

    # Delete the abalone_regression Model
    m = Model("abalone-regression")
    if m.exists():
        print("Deleting abalone-regression model...")
        m.delete()

    # Delete the abalone_regression Endpoint
    end = Endpoint("abalone-regression-end")
    if end.exists():
        print("Deleting abalone-regression-end endpoint...")
        end.delete()

    # Classification Artifacts
    fs = FeatureSet("abalone_classification")
    if fs.exists():
        print("Deleting abalone_classification...")
        fs.delete()
    m = Model("abalone-classification")
    if m.exists():
        print("Deleting abalone-classification model...")
        m.delete()
    end = Endpoint("abalone-classification-end")
    if end.exists():
        print("Deleting abalone-classification-end endpoint...")
        end.delete()

    time.sleep(5)
    print("All test artifacts should now be deleted!")
