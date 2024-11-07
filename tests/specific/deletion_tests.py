import pytest
import sageworks  # noqa: F401
import logging
from sageworks.utils.test_data_generator import TestDataGenerator
from sageworks.api import DataSource, FeatureSet, Model, Endpoint, ModelType

# Set the logging level
logging.getLogger("sageworks").setLevel(logging.DEBUG)


def create_data_source():
    test_data = TestDataGenerator()
    df = test_data.person_data()
    if not DataSource("abc").exists():
        DataSource(df, name="abc")


def create_feature_set():
    create_data_source()

    # If the feature set doesn't exist, create it
    if not FeatureSet("abc_features").exists():
        DataSource("abc").to_features("abc_features", id_column="id")


def create_model():
    create_feature_set()

    # If the model doesn't exist, create it
    if not Model("abc-regression").exists():
        FeatureSet("abc_features").to_model(
            model_type=ModelType.REGRESSOR, target_column="iq_score", name="abc-regression"
        )


def create_endpoint():
    create_model()

    # Create some new endpoints
    if not Endpoint("abc-end").exists():
        Model("abc-regression").to_endpoint(name="abc-end")


@pytest.mark.long
def test_endpoint_deletion():
    create_endpoint()

    # Now Delete the endpoint
    Endpoint("abc-end").delete()


@pytest.mark.long
def test_model_deletion():
    create_model()

    # Now Delete the Model
    Model("abc-regression").delete()


@pytest.mark.long
def test_feature_set_deletion():
    create_feature_set()

    # Now Delete the FeatureSet
    FeatureSet("abc_features").delete()


@pytest.mark.long
def test_data_source_deletion():
    create_data_source()

    # Now Delete the DataSource
    DataSource("abc").delete()


if __name__ == "__main__":

    test_endpoint_deletion()
    test_model_deletion()
    test_feature_set_deletion()
    test_data_source_deletion()
