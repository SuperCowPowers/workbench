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
    DataSource.delete("test_data")
    DataSource.delete("abc")
    DataSource.delete("abc_2")
    FeatureSet.delete("test_features")
    Model.delete("test-model")
    Endpoint.delete("test-end")
    Endpoint.delete("abc")
    Endpoint.delete("abc-2")

    # Delete the abalone_data Artifacts
    DataSource.delete("abalone_data")
    DataSource.delete("abalone_data_copy")
    FeatureSet.delete("abalone_features")
    Model.delete("abalone-regression")
    Endpoint.delete("abalone-regression-end")
    Endpoint.delete("abalone-regression-end-rt")

    # Delete abalone_classification Artifacts
    FeatureSet.delete("abalone_classification")
    Model.delete("abalone-classification")
    Endpoint.delete("abalone-classification-end")

    # Delete the wine_data Artifacts
    DataSource.delete("wine_data")
    FeatureSet.delete("wine_features")
    Model.delete("wine-classification")
    Endpoint.delete("wine-classification-end")

    # AQSol Artifacts
    DataSource.delete("aqsol_data")
    FeatureSet.delete("aqsol_features")
    Model.delete("aqsol-regression")
    Endpoint.delete("aqsol-regression-end")
    FeatureSet.delete("aqsol_mol_descriptors")
    Model.delete("aqsol-mol-regression")
    Endpoint.delete("aqsol-mol-regression-end")

    # AQSol Artifacts (Classification)
    Model.delete("aqsol-mol-class")
    Endpoint.delete("aqsol-mol-class-end")

    # Quantile Regressors
    Model.delete("abalone-quantile-reg")
    Endpoint.delete("abalone-qr-end")
    Model.delete("aqsol-quantile-reg")
    Endpoint.delete("aqsol-qr-end")

    # Scikit Learn Models
    Model.delete("abalone-knn-reg")
    Endpoint.delete("abalone-knn-end")
    Model.delete("abalone-clusters")
    Endpoint.delete("abalone-clusters-end")
    Model.delete("wine-rfc-class")
    Model.delete("aqsol-knn-reg")

    time.sleep(5)
    print("All test artifacts should now be deleted!")
