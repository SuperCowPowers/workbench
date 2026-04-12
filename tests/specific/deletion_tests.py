import pytest
import workbench  # noqa: F401
import logging
from workbench.utils.synthetic_data_generator import SyntheticDataGenerator
from workbench.api import DataSource, FeatureSet, Model, Endpoint, ModelType, ModelFramework

# Set the logging level
logging.getLogger("workbench").setLevel(logging.DEBUG)


def create_data_source():
    test_data = SyntheticDataGenerator()
    df = test_data.person_data()
    if not DataSource("delete_test").exists():
        DataSource(df, name="delete_test")


def create_feature_set():
    create_data_source()

    # If the feature set doesn't exist, create it
    if not FeatureSet("delete_test").exists():
        DataSource("delete_test").to_features("delete_test", id_column="id")


def create_model():
    create_feature_set()

    # If the model doesn't exist, create it
    if not Model("delete-test").exists():
        FeatureSet("delete_test").to_model(name="delete-test", model_type=ModelType.REGRESSOR, model_framework=ModelFramework.XGBOOST, target_column="iq_score")


def create_endpoint():
    create_model()

    # Create some new endpoints
    if not Endpoint("delete-test").exists():
        Model("delete-test").to_endpoint()


@pytest.mark.long
def test_endpoint_deletion():
    create_endpoint()

    # Now Delete the endpoint
    Endpoint("delete-test").delete()


@pytest.mark.long
def test_model_deletion():
    create_model()

    # Now Delete the Model
    Model("delete-test").delete()


@pytest.mark.long
def test_feature_set_deletion():
    create_feature_set()

    # Now Delete the FeatureSet
    FeatureSet("delete_test").delete()


@pytest.mark.long
def test_data_source_deletion():
    create_data_source()

    # Now Delete the DataSource
    DataSource("delete_test").delete()


if __name__ == "__main__":

    test_endpoint_deletion()
    test_model_deletion()
    test_feature_set_deletion()
    test_data_source_deletion()
