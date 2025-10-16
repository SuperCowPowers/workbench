import pandas as pd
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup
import time
import multiprocessing
import multiprocess

# Toggle this flag to test spawn mode fix
USE_SPAWN_MODE = True  # Set to True to fix the Tahoe hang issue

if __name__ == "__main__":
    if USE_SPAWN_MODE:
        print("Using SPAWN mode (fix for Tahoe)")
        multiprocessing.set_start_method("spawn", force=True)
        multiprocess.set_start_method("spawn", force=True)
    else:
        print("Using default fork mode (will hang on Tahoe)")

    # Create fake data
    data = pd.DataFrame(
        {
            "record_id": [f"id_{i}" for i in range(10)],
            "feature_1": [float(i) for i in range(10)],
            "feature_2": [float(i * 2) for i in range(10)],
            "event_time": [time.time()] * 10,
        }
    )

    # Setup SageMaker session
    sagemaker_session = sagemaker.Session()

    # Define feature group
    feature_group_name = "temp_delete_me"
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session)

    # Create feature definitions
    feature_group.load_feature_definitions(data_frame=data)

    # Create feature group
    print("Creating feature group...")
    feature_group.create(
        s3_uri=f"s3://{sagemaker_session.default_bucket()}/featurestore",
        record_identifier_name="record_id",
        event_time_feature_name="event_time",
        role_arn=sagemaker.get_execution_role(),
        enable_online_store=True,
    )

    # Wait for feature group to be created (can take 1-2 minutes)
    print("Waiting for feature group to be ready...")
    status = feature_group.describe().get("FeatureGroupStatus")
    while status == "Creating":
        print(f"Status: {status}... waiting 10 seconds")
        time.sleep(10)
        status = feature_group.describe().get("FeatureGroupStatus")
    print(f"Feature group status: {status}")

    # This will hang on macOS Tahoe with USE_SPAWN_MODE=False
    print("Starting ingest...")
    feature_group.ingest(data_frame=data, max_workers=2, max_processes=2, wait=True)
    print("Ingest completed!")
