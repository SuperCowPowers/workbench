"""Timing of pipeline endpoints vs. individual model endpoints"""

import time

# Workbench imports
from workbench.api import FeatureSet, Endpoint

feature_set_name = "aqsol_features"
id_column = "id"

# Pull in a dataframe of smiles
fs = FeatureSet(feature_set_name)
df = fs.pull_dataframe()
total_df = df[[id_column, "smiles"]]
print(total_df.shape)
print(total_df.head())

# We're going to run 3 timings on 10, 100, 500, 1000, 10000 rows
for n in [10, 100]:
    input_df = total_df.head(n)
    print(f"Timing for {n} rows")

    # First time the individual endpoints

    # Tautomerize
    total_start = time.time()
    end_1 = Endpoint("tautomerize-v0-rt")
    df = end_1.fast_inference(input_df)
    time_taut = time.time() - total_start

    # Molecular Descriptors
    start = time.time()
    end_2 = Endpoint("smiles-to-md-v0-rt")
    df = end_2.fast_inference(df)
    time_md = time.time() - start

    # AQSOL Classification
    start = time.time()
    end_3 = Endpoint("aqsol-mol-class-rt")
    df = end_3.fast_inference(df)
    time_model = time.time() - start

    print(f"Individual endpoints Total: {time_model - total_start} seconds")
    print(f"Taut: {time_taut} seconds, MD: {time_md} seconds, Model: {time_model} seconds")

    # Now time the pipeline endpoint
    start = time.time()
    end = Endpoint("pipeline-model")
    df = end.fast_inference(input_df)
    end = time.time()
    print(f"Pipeline endpoint: {end - start} seconds")
