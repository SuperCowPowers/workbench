"""This Script creates the SageWorks Artifacts in AWS needed for the tests
   
   DataSources:
       - test_data
       - abalone_data
   FeatureSets:
       - test_feature_set
       - abalone_feature_set
"""
import sys
from pathlib import Path
from sageworks.transforms.data_loaders.light.csv_to_data_source import CSVToDataSource
from sageworks.transforms.data_to_features.light.data_to_features_light import DataToFeaturesLight

if __name__ == "__main__":
    # Get the path to the dataset in the repository data directory
    test_data_path = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "test_data.csv"
    abalone_data_path = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "abalone.csv"

    # Create the test_data DataSource
    my_loader = CSVToDataSource(test_data_path, "test_data")
    my_loader.set_output_tags("test:small")
    my_loader.transform()

    # Create the abalone_data DataSource
    my_loader = CSVToDataSource(abalone_data_path, "abalone_data")
    my_loader.set_output_tags("abalone:public")
    my_loader.transform()

    # Create the test_feature_set FeatureSet
    data_to_features = DataToFeaturesLight("test_data", "test_feature_set")
    data_to_features.set_output_tags(["test", "small"])
    data_to_features.transform(id_column="id", event_time_column="date")

    # Create the abalone_feature_set FeatureSet
    data_to_features = DataToFeaturesLight("abalone_data", "abalone_feature_set")
    data_to_features.set_output_tags(["abalone", "public"])
    data_to_features.transform()
