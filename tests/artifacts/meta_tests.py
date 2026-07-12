"""Tests for the Meta API singular getters.

The list methods (models(), feature_sets(), ...) get exercised all over the
test suite, but the singular getters go through a Meta -> CloudMeta -> AWSMeta
super() chain where each layer renames the keyword argument. A mismatch in
that chain is a guaranteed TypeError (regression: feature_set() and model()
passed their grandparent's keyword names to their parent).
"""

# Workbench Imports
from workbench.api.meta import Meta


def test_data_source():
    """Singular data_source() getter returns details dict"""
    meta = Meta()
    details = meta.data_source("abalone_data")
    assert isinstance(details, dict)


def test_feature_set():
    """Singular feature_set() getter returns details dict"""
    meta = Meta()
    details = meta.feature_set("abalone_features")
    assert isinstance(details, dict)


def test_model():
    """Singular model() getter returns details dict"""
    meta = Meta()
    details = meta.model("abalone-regression")
    assert isinstance(details, dict)


def test_endpoint():
    """Singular endpoint() getter returns details dict"""
    meta = Meta()
    details = meta.endpoint("abalone-regression")
    assert isinstance(details, dict)


def test_not_found_returns_none():
    """All the singular getters return None for nonexistent artifacts"""
    meta = Meta()
    assert meta.feature_set("does-not-exist") is None
    assert meta.model("does-not-exist") is None
    assert meta.endpoint("does-not-exist") is None


if __name__ == "__main__":
    test_data_source()
    test_feature_set()
    test_model()
    test_endpoint()
    test_not_found_returns_none()
    print("All Meta singular getter tests passed!")
