"""Invoke a SMILES-based endpoint with boto3's sagemaker-runtime client (no Workbench dependency).

Credentials and region come from the standard AWS chain (AWS_PROFILE, env vars, instance role).
"""

from io import StringIO

import boto3
import pandas as pd

endpoint_name = "all-regression-1"


def invoke_endpoint_csv(endpoint_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Send a CSV payload to the endpoint and return the CSV response as a DataFrame."""
    runtime = boto3.Session().client("sagemaker-runtime")
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=df.to_csv(index=False),
        ContentType="text/csv",
        Accept="text/csv",
    )
    return pd.read_csv(StringIO(response["Body"].read().decode("utf-8")))


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "id": ["aspirin", "caffeine", "ibuprofen", "paracetamol", "nicotine"],
            "smiles": [
                "CC(=O)OC1=CC=CC=C1C(=O)O",
                "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                "CC(=O)NC1=CC=C(C=C1)O",
                "CN1CCCC1C2=CN=CC=C2",
            ],
        }
    )

    print("Valid SMILES...")
    print(invoke_endpoint_csv(endpoint_name, df))

    print("\nOne invalid SMILES...")
    bad_df = df.copy()
    bad_df.loc[2, "smiles"] = "not_a_smiles"
    print(invoke_endpoint_csv(endpoint_name, bad_df))

    print("\nSingle invalid SMILES...")
    single_bad_df = pd.DataFrame({"id": ["bad"], "smiles": ["not_a_smiles"]})
    print(invoke_endpoint_csv(endpoint_name, single_bad_df))
