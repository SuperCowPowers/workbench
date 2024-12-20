import sys
import time
import logging
from io import StringIO
import matplotlib.pyplot as plt

# SageMaker imports
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from sagemaker import Predictor

# Workbench imports
from workbench.api import FeatureSet, Model, Endpoint

log = logging.getLogger("workbench")


def plot_timings(timings):
    """Plots the inference timings."""
    x = [1, 10, 100, 1000, 10000]
    plt.figure(figsize=(10, 6))

    plt.plot(x, timings["Workbench Serverless"], label="Workbench Serverless", color="blue")
    plt.plot(x, timings["Workbench Realtime"], label="Workbench Realtime", color="green")
    plt.plot(x, timings["AWS Serverless"], label="AWS Serverless", color="red")
    plt.plot(x, timings["AWS Realtime"], label="AWS Realtime", color="orange")

    plt.xscale("log")
    plt.xlabel("Number of Rows")
    plt.ylabel("Inference Time (seconds)")
    plt.title("Inference Timing Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    plot = True  # Set this flag to control plotting

    # Check that the two Endpoints exist
    if not Endpoint("test-timing-serverless").exists():
        log.error("The test-timing-serverless endpoint does not exist. Please create it first.")
        sys.exit(1)

    if not Endpoint("test-timing-realtime").exists():
        log.error("The test-timing-realtime endpoint does not exist. Please create it first.")
        sys.exit(1)

    # Timing results storage
    timings = {
        "Workbench Serverless": [],
        "Workbench Realtime": [],
        "AWS Serverless": [],
        "AWS Realtime": [],
    }

    # Get the endpoints
    start_time = time.time()
    serverless = Endpoint("test-timing-serverless")
    print(f"Workbench Serverless Endpoint: Construction Time: {time.time() - start_time}")

    start_time = time.time()
    realtime = Endpoint("test-timing-realtime")
    print(f"Workbench Realtime Endpoint: Construction Time: {time.time() - start_time}")

    # Backtrace the endpoints to get the FeatureSet
    fs = FeatureSet(Model(serverless.get_input()).get_input())

    # Grab the FeatureSet data
    data = fs.pull_dataframe()

    #
    # Done with the data pull (we're not including data pull as part of the timing)
    #

    # Now we're going to skip any Workbench code and just use the AWS SDK
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
        aws_serverless_time = time.time() - start_time
        timings["AWS Serverless"].append(aws_serverless_time)
        print(f"\tServerless Inference Time: {aws_serverless_time}")

        # Convert into a CSV buffer and send it to the AWS realtime predictor
        start_time = time.time()
        csv_buffer = StringIO()
        data_sample.to_csv(csv_buffer, index=False)
        realtime_predictor.predict(csv_buffer.getvalue())
        aws_realtime_time = time.time() - start_time
        timings["AWS Realtime"].append(aws_realtime_time)
        print(f"\tRealtime Inference Time: {aws_realtime_time}")

    print("\n*** Timing Inference using Workbench API ***")

    # Collect inference times on 1, 10, 100, 1000, and 10000 rows
    for i in [1, 10, 100, 1000, 10000]:
        data_sample = data.sample(n=i, replace=True)
        print(f"\nTiming Inference on {len(data_sample)} rows")
        start_time = time.time()
        serverless.fast_inference(data_sample)
        serverless_time = time.time() - start_time
        timings["Workbench Serverless"].append(serverless_time)
        print(f"\tServerless Inference Time: {serverless_time}")

        start_time = time.time()
        realtime.fast_inference(data_sample)
        realtime_time = time.time() - start_time
        timings["Workbench Realtime"].append(realtime_time)
        print(f"\tRealtime Inference Time: {realtime_time}")

    # Plot timings if the flag is set
    if plot:
        plot_timings(timings)
