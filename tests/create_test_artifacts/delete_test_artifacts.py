"""This Script Deletes the SageWorks Artifacts used for the tests"""

import time
from sageworks.api.data_source import DataSource
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model
from sageworks.api.endpoint import Endpoint


if __name__ == "__main__":

    # Delete the test_data Artifacts
    DataSource.managed_delete("test_data")
    DataSource.managed_delete("abc")
    DataSource.managed_delete("abc_2")
    FeatureSet.managed_delete("test_features")
    Endpoint.managed_delete("abc")
    Endpoint.managed_delete("abc-2")

    # Delete the abalone_data Artifacts
    DataSource.managed_delete("abalone_data")
    DataSource.managed_delete("abalone_data_copy")
    FeatureSet.managed_delete("abalone_features")
    Model.managed_delete("abalone-regression")
    Endpoint.managed_delete("abalone-regression-end")
    Endpoint.managed_delete("abalone-regression-end-rt")

    # Delete abalone_classification Artifacts
    FeatureSet.managed_delete("abalone_classification")
    Model.managed_delete("abalone-classification")
    Endpoint.managed_delete("abalone-classification-end")

    # Delete the wine_data Artifacts
    DataSource.managed_delete("wine_data")
    FeatureSet.managed_delete("wine_features")
    Model.managed_delete("wine-classification")
    Endpoint.managed_delete("wine-classification-end")

    # AQSol Artifacts
    DataSource.managed_delete("aqsol_data")
    FeatureSet.managed_delete("aqsol_features")
    Model.managed_delete("aqsol-regression")
    Endpoint.managed_delete("aqsol-regression-end")
    FeatureSet.managed_delete("aqsol_mol_descriptors")
    Model.managed_delete("aqsol-mol-regression")
    Endpoint.managed_delete("aqsol-mol-regression-end")

    # AQSol Artifacts (Classification)
    Model.managed_delete("aqsol-mol-class")
    Endpoint.managed_delete("aqsol-mol-class-end")

    # Quantile Regressors
    Model.managed_delete("abalone-quantile-reg")
    Endpoint.managed_delete("abalone-qr-end")
    Model.managed_delete("aqsol-quantile-reg")
    Endpoint.managed_delete("aqsol-qr-end")

    # Scikit Learn Models
    Model.managed_delete("abalone-knn-reg")
    Endpoint.managed_delete("abalone-knn-end")
    Model.managed_delete("abalone-clusters")
    Endpoint.managed_delete("abalone-clusters-end")
    Model.managed_delete("wine-rfc-class")
    Model.managed_delete("aqsol-knn-reg")

    time.sleep(5)
    print("All test artifacts should now be deleted!")
