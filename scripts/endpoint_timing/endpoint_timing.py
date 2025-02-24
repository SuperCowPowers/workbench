"""Timing of pipeline endpoints vs. individual model endpoints"""

import re
import time

# Workbench imports
from workbench.api import FeatureSet, Endpoint
from workbench.utils.fast_inference import fast_inference

feature_set_name = "aqsol_features"
id_column = "id"

# Pull in a dataframe of smiles
fs = FeatureSet(feature_set_name)
df = fs.pull_dataframe()
total_df = df[[id_column, "smiles"]]

# Remove any rows with this SMILES substring "[I-].CN1C=CC=C\C1=C/[NH+]=O"
print(f"Before: {total_df.shape}")
# total_df = total_df[~total_df["smiles"].str.contains(r"[I-].CN1C=CC=C\C1=C/[NH+]=O")]
pattern = re.escape(r"[I-].CN1C=CC=C\C1=C/[NH+]=O")  # Escapes all special characters
total_df = total_df[~total_df["smiles"].str.contains(pattern)]
print(f"After: {total_df.shape}")

# First let's dump the information about all the endpoints
end_1 = Endpoint("tautomerize-v0-rt")
print(f"Endpoint: {end_1.uuid}, Instance: {end_1.instance_type}")
end_2 = Endpoint("smiles-to-md-v0-rt")
print(f"Endpoint: {end_2.uuid}, Instance: {end_2.instance_type}")
end_3 = Endpoint("aqsol-mol-class-rt")
print(f"Endpoint: {end_3.uuid}, Instance: {end_3.instance_type}")
end_pipe = Endpoint("pipeline-model")
print(f"Endpoint: {end_pipe.uuid}, Instance: {end_pipe.instance_type}")
end_pipe_fast = Endpoint("pipeline-model-fast")
print(f"Endpoint: {end_pipe_fast.uuid}, Instance: {end_pipe_fast.instance_type}")

# We're going to grab our Sagemaker Session from an endpoint (the all give the same session)
session = end_1.sm_session

# We're going to run 3 timings on 10, 100, 500, 1000, 10000 rows
for n in [10, 100, 500, 1000, 10000]:
    input_df = total_df.head(n)
    print(f"Timing for {n} rows")

    # First time the individual endpoints

    # Tautomerize
    total_start = time.time()
    # end_1 = Endpoint("tautomerize-v0-rt")
    # df = end_1.fast_inference(input_df)
    df = fast_inference(end_1.uuid, input_df, session)
    time_taut = time.time() - total_start

    # Molecular Descriptors
    start = time.time()
    # end_2 = Endpoint("smiles-to-md-v0-rt")
    # df = end_2.fast_inference(df)
    df = fast_inference(end_2.uuid, df, session)
    time_md = time.time() - start

    # AQSOL Classification
    start = time.time()
    # end_3 = Endpoint("aqsol-mol-class-rt")
    # df = end_3.fast_inference(df)
    df = fast_inference(end_3.uuid, df, session)
    time_model = time.time() - start
    total_time = time.time() - total_start

    print(f"Individual endpoints Total: {total_time} seconds")
    print(f"Taut: {time_taut} seconds, MD: {time_md} seconds, Model: {time_model} seconds")

    # Now time the pipeline endpoint
    start = time.time()
    # end_pipe = Endpoint("pipeline-model")
    # df = end_pipe.fast_inference(input_df)
    df = fast_inference(end_pipe.uuid, input_df, session)
    end = time.time()
    print(f"Pipeline endpoint: {end - start} seconds")

    # Now time the pipeline fast endpoint
    start = time.time()
    # end_pipe_fast = Endpoint("pipeline-model-fast")
    # df = end_pipe_fast.fast_inference(input_df)
    df = fast_inference(end_pipe_fast.uuid, input_df, session)
    end = time.time()
    print(f"Pipeline Fast endpoint: {end - start} seconds")
