"""This Script Deletes the SageWorks Artifacts used for the tests"""

import time
from sageworks.api.data_source import DataSource
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model
from sageworks.api.endpoint import Endpoint
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker


if __name__ == "__main__":
    # This forces a refresh on all the data we get from the AWs Broker
    AWSServiceBroker().get_all_metadata(force_refresh=True)

    # Delete the test_data Artifacts
    ds = DataSource("test_data")
    if ds.exists():
        print("Deleting test_data...")
        ds.delete()
    fs = FeatureSet("test_features")
    if fs.exists():
        print("Deleting test_features...")
        fs.delete()
    m = Model("test-model")
    if m.exists():
        print("Deleting test-model model...")
        m.delete()
    end = Endpoint("test-end")
    if end.exists():
        print("Deleting test-end endpoint...")
        end.delete()

    # Delete the abalone_data Artifacts
    ds = DataSource("abalone_data")
    if ds.exists():
        print("Deleting abalone_data...")
        ds.delete()
    ds = DataSource("abalone_data_copy")
    if ds.exists():
        print("Deleting abalone_data_copy...")
        ds.delete()
    fs = FeatureSet("abalone_features")
    if fs.exists():
        print("Deleting abalone_features...")
        fs.delete()
    m = Model("abalone-regression")
    if m.exists():
        print("Deleting abalone-regression model...")
        m.delete()
    end = Endpoint("abalone-regression-end")
    if end.exists():
        print("Deleting abalone-regression-end...")
        end.delete()
    end = Endpoint("abalone-regression-end-rt")
    if end.exists():
        print("Deleting abalone-regression-end-rt...")
        end.delete()

    # Delete abalone_classification Artifacts
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

    # Delete the wine_data Artifacts
    ds = DataSource("wine_data")
    if ds.exists():
        print("Deleting wine_data...")
        ds.delete()
    fs = FeatureSet("wine_features")
    if fs.exists():
        print("Deleting wine_features...")
        fs.delete()
    m = Model("wine-classification")
    if m.exists():
        print("Deleting wine-classification model...")
        m.delete()
    end = Endpoint("wine-classification-end")
    if end.exists():
        print("Deleting wine-classification-end endpoint...")
        end.delete()

    # AQSol Artifacts
    ds = DataSource("aqsol_data")
    if ds.exists():
        print("Deleting aqsol_data...")
        ds.delete()
    fs = FeatureSet("aqsol_features")
    if fs.exists():
        print("Deleting aqsol_features...")
        fs.delete()
    m = Model("aqsol-regression")
    if m.exists():
        print("Deleting aqsol-regression model...")
        m.delete()
    end = Endpoint("aqsol-regression-end")
    if end.exists():
        print("Deleting aqsol-regression-end endpoint...")
        end.delete()
    fs = FeatureSet("aqsol_mol_descriptors")
    if fs.exists():
        print("Deleting aqsol_mol_descriptors...")
        fs.delete()
    m = Model("aqsol-descriptors-regression")
    if m.exists():
        print("Deleting aqsol-mol-regression model...")
        m.delete()
    end = Endpoint("aqsol-mol-regression-end")
    if end.exists():
        print("Deleting aqsol-mol-regression-end endpoint...")
        end.delete()

    time.sleep(5)
    print("All test artifacts should now be deleted!")
