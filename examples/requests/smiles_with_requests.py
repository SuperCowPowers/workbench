"""Invoke a SMILES-based endpoint with plain `requests` (no Workbench dependency).

Requests are SigV4-signed with botocore (ships with boto3). Credentials and
region come from the standard AWS chain (AWS_PROFILE, env vars, instance role).
"""

from io import StringIO

import boto3
import pandas as pd
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

endpoint_name = "all-regression-1"


def invoke_endpoint_csv(endpoint_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """POST a signed CSV payload to the endpoint and return the CSV response as a DataFrame."""
    session = boto3.Session()
    url = f"https://runtime.sagemaker.{session.region_name}.amazonaws.com/endpoints/{endpoint_name}/invocations"
    headers = {"Content-Type": "text/csv", "Accept": "text/csv"}
    payload = df.to_csv(index=False)

    # Sign the request, then send it with requests
    aws_request = AWSRequest(method="POST", url=url, data=payload, headers=headers)
    SigV4Auth(session.get_credentials(), "sagemaker", session.region_name).add_auth(aws_request)
    response = requests.post(url, data=payload, headers=dict(aws_request.headers), timeout=90)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


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
