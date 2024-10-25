from concurrent.futures import ProcessPoolExecutor
from sageworks.api import DataSource, FeatureSet, Model, ModelType, Endpoint, Meta
from sageworks.utils.test_data_generator import TestDataGenerator


# Function to create a model pipeline for a given model name
def create_model_pipeline(model_name):
    print(f"Creating pipeline for model: {model_name}")

    # This forces a refresh on all the data we get from the AWS Broker
    Meta().refresh_all_aws_meta()

    # Create the test_data DataSource
    if not DataSource("test_data").exists():
        # Create a new Data Source from a dataframe of test data
        test_data = TestDataGenerator()
        df = test_data.person_data()
        DataSource(df, name="test_data")

    # Convert model name to a FeatureSet name
    feature_set_name = f"{model_name.replace('-', '_')}_features"
    # Create the test_features FeatureSet
    if not FeatureSet(feature_set_name).exists():
        ds = DataSource("test_data")
        ds.to_features(feature_set_name, id_column="id", event_time_column="date")

    # Create the model with the given name
    if not Model(model_name).exists():
        fs = FeatureSet(feature_set_name)
        m = fs.to_model(
            model_type=ModelType.REGRESSOR,
            name=model_name,
            target_column="iq_score",
            tags=["test"],
            description=f"Test Model {model_name}",
        )
        m.set_owner("test")

    # Create the Endpoint for the model
    endpoint_name = f"{model_name}-end"
    if not Endpoint(endpoint_name).exists():
        m = Model(model_name)
        end = m.to_endpoint(endpoint_name, tags=["test"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)


# List of model names to run in parallel
model_names = [f"test-model-{i}" for i in range(1)]

if __name__ == "__main__":
    # Use ProcessPoolExecutor to run in parallel
    with ProcessPoolExecutor(max_workers=16) as executor:
        # Submit parallel jobs
        executor.map(create_model_pipeline, model_names)
