"""Endpoints Timing Script

Endpoints:
    - test-timing-serverless, test-timing-realtime (Model: abalone-regression)
"""

import sys
import time
import logging
from io import StringIO
import pandas as pd

# SageMaker imports
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from sagemaker import Predictor

# SageWorks imports
from sageworks.api import FeatureSet, Model, Endpoint

log = logging.getLogger("sageworks")


if __name__ == "__main__":

    # Check that the two Endpoints exist
    if not Endpoint("test-timing-serverless").exists():
        log.error("The test-timing-serverless endpoint does not exist. Please create it first.")
        sys.exit(1)

    if not Endpoint("test-timing-realtime").exists():
        log.error("The test-timing-realtime endpoint does not exist. Please create it first.")
        sys.exit(1)

    # Get the endpoints
    start_time = time.time()
    serverless = Endpoint("test-timing-serverless")
    print(f"SageWorks Serverless Endpoint: Construction Time: {time.time() - start_time}")

    start_time = time.time()
    realtime = Endpoint("test-timing-realtime")
    print(f"SageWorks Realtime Endpoint: Construction Time: {time.time() - start_time}")

    # Backtrace the endpoints to get the FeatureSet
    fs = FeatureSet(Model(serverless.get_input()).get_input())

    # Grab the FeatureSet data
    data = fs.pull_dataframe()

    print("\n*** Timing Inference using SageWorks API ***")

    # Collect inference times on 1, 10, 100, 1000, and 10000 rows
    for i in [1, 10, 100, 1000, 10000]:
        data_sample = data.sample(n=i, replace=True)
        print(f"\nTiming Inference on {len(data_sample)} rows")
        start_time = time.time()
        serverless.fast_inference(data_sample)
        print(f"\tServerless Inference Time: {time.time() - start_time}")
        start_time = time.time()
        realtime.fast_inference(data_sample)
        print(f"\tRealtime Inference Time: {time.time() - start_time}")

    # Now we're going to skip any SageWorks code and just use the AWS SDK
    print("\n*** Timing Inference using JUST the AWS SDK ***")

    # Predictor creation
    start_time = time.time()
    serverless_predictor = Predictor(
        "test-timing-serverless",
        sagemaker_session=serverless.sm_session,
        serializer=CSVSerializer(),
        deserializer=CSVDeserializer(),
    )
    realtime_predictor = Predictor(
        "test-timing-realtime",
        sagemaker_session=realtime.sm_session,
        serializer=CSVSerializer(),
        deserializer=CSVDeserializer(),
    )
    print(f"Sagemaker Predictor: Construction Time: {time.time() - start_time}")

    # Just AWS: Collect inference times on 1, 10, 100, 1000, and 10000 rows
    for i in [1, 10, 100, 1000, 10000]:
        data_sample = data.sample(n=i, replace=True)
        print(f"\nTiming Inference on {len(data_sample)} rows")

        # Convert into a CSV buffer and send it to the AWS serverless predictor
        start_time = time.time()
        csv_buffer = StringIO()
        data_sample.to_csv(csv_buffer, index=False)
        serverless_predictor.predict(csv_buffer.getvalue())
        print(f"\tServerless Inference Time: {time.time() - start_time}")

        # Convert into a CSV buffer and send it to the AWS realtime predictor
        start_time = time.time()
        csv_buffer = StringIO()
        data_sample.to_csv(csv_buffer, index=False)
        realtime_predictor.predict(csv_buffer.getvalue())
        print(f"\tRealtime Inference Time: {time.time() - start_time}")

    # Storage
    # Construct a DataFrame from the results
    # results_df = pd.DataFrame.from_records(results[1:], columns=results[0])
    # print("\n\nPrediction Results DataFrame:")
    # print(results_df.head())
