"""Invoke a Workbench endpoint with plain `requests` (no Workbench dependency).

Requests are SigV4-signed with botocore (ships with boto3). Credentials and
region come from the standard AWS chain (AWS_PROFILE, env vars, instance role).
Endpoints accept and return both CSV and JSON.
"""

from io import StringIO

import boto3
import pandas as pd
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

endpoint_name = "abalone-regression"


def invoke_endpoint(endpoint_name: str, payload: str, content_type: str) -> requests.Response:
    """POST a signed payload to the endpoint; request and response use the same content type."""
    session = boto3.Session()
    url = f"https://runtime.sagemaker.{session.region_name}.amazonaws.com/endpoints/{endpoint_name}/invocations"
    headers = {"Content-Type": content_type, "Accept": content_type}

    # Sign the request, then send it with requests
    aws_request = AWSRequest(method="POST", url=url, data=payload, headers=headers)
    SigV4Auth(session.get_credentials(), "sagemaker", session.region_name).add_auth(aws_request)
    response = requests.post(url, data=payload, headers=dict(aws_request.headers), timeout=90)
    response.raise_for_status()
    return response


def invoke_endpoint_csv(endpoint_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Invoke the endpoint with CSV input and output."""
    response = invoke_endpoint(endpoint_name, df.to_csv(index=False), "text/csv")
    return pd.read_csv(StringIO(response.text))


def invoke_endpoint_json(endpoint_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Invoke the endpoint with JSON input and output."""
    response = invoke_endpoint(endpoint_name, df.to_json(orient="records"), "application/json")
    return pd.DataFrame(response.json())


if __name__ == "__main__":
    # A few abalone rows (replace with your own DataFrame)
    df = pd.DataFrame(
        {
            "sex": ["M", "M", "F"],
            "length": [0.455, 0.35, 0.53],
            "diameter": [0.365, 0.265, 0.42],
            "height": [0.095, 0.09, 0.135],
            "whole_weight": [0.514, 0.2255, 0.677],
            "shucked_weight": [0.2245, 0.0995, 0.2565],
            "viscera_weight": [0.101, 0.0485, 0.1415],
            "shell_weight": [0.15, 0.07, 0.21],
        }
    )

    print("CSV request/response...")
    csv_df = invoke_endpoint_csv(endpoint_name, df)
    print(csv_df.head())

    print("\nJSON request/response...")
    json_df = invoke_endpoint_json(endpoint_name, df)
    print(json_df.head())
