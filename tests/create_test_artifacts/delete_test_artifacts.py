"""This Script Deletes the Workbench Artifacts used for the tests"""

import time
from workbench.api.data_source import DataSource
from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model
from workbench.api.endpoint import Endpoint


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
    Endpoint.managed_delete("abalone-regression")
    Endpoint.managed_delete("abalone-regression-end-rt")

    # Delete abalone_classification Artifacts
    FeatureSet.managed_delete("abalone_classification")
    Model.managed_delete("abalone-classification")
    Endpoint.managed_delete("abalone-classification")

    # Delete the wine_data Artifacts
    DataSource.managed_delete("wine_data")
    FeatureSet.managed_delete("wine_features")
    Model.managed_delete("wine-classification")
    Endpoint.managed_delete("wine-classification")

    # AQSol Artifacts
    DataSource.managed_delete("aqsol_data")
    FeatureSet.managed_delete("aqsol_features")
    Model.managed_delete("aqsol-regression")
    Endpoint.managed_delete("aqsol-regression")
    FeatureSet.managed_delete("aqsol_mol_descriptors")
    Model.managed_delete("aqsol-mol-regression")
    Endpoint.managed_delete("aqsol-mol-regression")

    # AQSol Artifacts (Classification)
    Model.managed_delete("aqsol-mol-class")
    Endpoint.managed_delete("aqsol-mol-class")

    # Quantile Regressors
    Model.managed_delete("abalone-quantile-reg")
    Endpoint.managed_delete("abalone-qr")
    Model.managed_delete("aqsol-quantile-reg")
    Endpoint.managed_delete("aqsol-qr")

    # Scikit Learn Models
    Model.managed_delete("abalone-knn-reg")
    Endpoint.managed_delete("abalone-knn")
    Model.managed_delete("abalone-clusters")
    Endpoint.managed_delete("abalone-clusters")
    Model.managed_delete("wine-rfc-class")
    Model.managed_delete("aqsol-knn-reg")

    time.sleep(5)
    print("All test artifacts should now be deleted!")
