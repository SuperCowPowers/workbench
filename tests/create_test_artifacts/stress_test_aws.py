from concurrent.futures import ProcessPoolExecutor
from sageworks.api import DataSource, FeatureSet, Model, ModelType, Endpoint, Meta
from sageworks.utils.test_data_generator import TestDataGenerator

parallel_jobs = 4

# Define expected artifacts
expected_artifacts = {
    "feature_sets": [f"stress_features_{i}" for i in range(parallel_jobs)],
    "models": [f"stress-model-{i}" for i in range(parallel_jobs)],
    "endpoints": [f"stress-end-{i}" for i in range(parallel_jobs)],
}


# Function to create a model pipeline for a given model name
def create_model_pipeline(model_name):
    print(f"Creating pipeline for model: {model_name}")

    # Create the test_data DataSource if it doesn't already exist
    if not DataSource("test_data").exists():
        test_data = TestDataGenerator()
        df = test_data.person_data()
        DataSource(df, name="test_data")

    # Convert model name to a FeatureSet name
    feature_set_name = model_name.replace("-model-", "_features_")
    # Create the FeatureSet if it doesn't already exist
    if not FeatureSet(feature_set_name).exists():
        ds = DataSource("test_data")
        ds.to_features(feature_set_name, id_column="id", event_time_column="date")

    # Create the Model if it doesn't already exist
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

    # Create the Endpoint for the Model if it doesn't already exist
    endpoint_name = model_name.replace("model", "end")
    if not Endpoint(endpoint_name).exists():
        m = Model(model_name)
        end = m.to_endpoint(endpoint_name, tags=["test"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)


if __name__ == "__main__":
    """A stress test to create AWS ML pipelines in parallel."""

    # List of model names to run in parallel
    model_names = expected_artifacts["models"]

    # This forces a refresh on all the data we get from the AWS Broker
    meta = Meta()
    meta.feature_sets(refresh=True)
    meta.models(refresh=True)
    meta.endpoints(refresh=True)

    # Use ProcessPoolExecutor to run in parallel
    with ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
        executor.map(create_model_pipeline, model_names)

    # Verify all expected artifacts
    meta = Meta()
    feature_sets = meta.feature_sets(refresh=True)
    models = meta.models(refresh=True)
    endpoints = meta.endpoints(refresh=True)
    missing_artifacts = {
        "feature_sets": [fs for fs in expected_artifacts["feature_sets"] if fs not in feature_sets],
        "models": [m for m in expected_artifacts["models"] if m not in models],
        "endpoints": [e for e in expected_artifacts["endpoints"] if e not in endpoints],
    }

    # Display missing artifacts
    if any(missing_artifacts.values()):
        print("Missing artifacts:", missing_artifacts)
    else:
        print("All expected artifacts created successfully.")
