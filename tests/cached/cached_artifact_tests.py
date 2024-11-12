"""Tests for the Cached Artifacts"""

from pprint import pprint

# SageWorks Imports
from sageworks.cached.cached_data_source import CachedDataSource
from sageworks.cached.cached_feature_set import CachedFeatureSet
from sageworks.cached.cached_model import CachedModel
from sageworks.cached.cached_endpoint import CachedEndpoint


def test_cached_data_source():
    print("\n\n*** Cached DataSource ***")
    my_data = CachedDataSource("abalone_data")
    pprint(my_data.summary())
    pprint(my_data.details())
    pprint(my_data.health_check())
    pprint(my_data.sageworks_meta())


def test_cached_feature_set():
    print("\n\n*** Cached FeatureSet ***")
    my_features = CachedFeatureSet("abalone_features")
    pprint(my_features.summary())
    pprint(my_features.details())
    pprint(my_features.health_check())
    pprint(my_features.sageworks_meta())


def test_cached_model():
    print("\n\n*** Cached Model ***")
    my_model = CachedModel("abalone-regression")
    pprint(my_model.summary())
    pprint(my_model.details())
    pprint(my_model.health_check())
    pprint(my_model.sageworks_meta())


def test_cached_endpoint():
    print("\n\n*** Cached Endpoint ***")
    my_endpoint = CachedEndpoint("abalone-regression-end")
    pprint(my_endpoint.summary())
    pprint(my_endpoint.details())
    pprint(my_endpoint.health_check())
    pprint(my_endpoint.sageworks_meta())


if __name__ == "__main__":

    # Run the tests
    test_cached_data_source()
    test_cached_feature_set()
    test_cached_model()
    test_cached_endpoint()

    # Shutdown the ThreadPoolExecutor (note: users should NOT call this)
    CachedEndpoint._shutdown()
