"""AWSSecretsManager: Manages Workbench secrets in AWS Secrets Manager."""

from typing import Union
import logging
import json
from botocore.exceptions import ClientError

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_session import AWSSession


class AWSSecretsManager:
    """AWSSecretsManager: Manages Workbench secrets in AWS Secrets Manager.

    Common Usage:
        ```python
        secrets = AWSSecretsManager()

        # List Secrets
        secrets.list()

        ['/workbench/api_key',
         '/workbench/database_credentials',
         '/workbench/oauth_token',
         '/workbench/pipelines/service_account']

        # Add Secret
        secrets.upsert("api_key", "my-secret-key")
        value = secrets.get("api_key")

        # Add structured data
        credentials = {"username": "admin", "password": "secret123", "host": "db.example.com"}
        secrets.upsert("database_credentials", credentials)

        # Retrieve secret
        db_creds = secrets.get("database_credentials")
        print(db_creds)

        {'username': 'admin', 'password': 'secret123', 'host': 'db.example.com'}

        # Delete secrets
        secrets.delete("database_credentials")
        ```
    """

    def __init__(self):
        """AWSSecretsManager Init Method"""
        self.log = logging.getLogger("workbench")

        # Initialize a Workbench Session (to assume the Workbench ExecutionRole)
        self.boto3_session = AWSSession().boto3_session

        # Create a Secrets Manager client
        self.secrets_client = self.boto3_session.client("secretsmanager")

    def list(self, prefix: str = None) -> list:
        """List all secrets in AWS Secrets Manager, optionally filtering by a prefix.

        Args:
            prefix (str, optional): A prefix to filter the secrets by. Defaults to None.

        Returns:
            list: A list of secret names.
        """
        try:
            # Set up parameters for the query
            params = {"MaxResults": 100}

            # Initialize the list to collect secret names
            all_secrets = []

            # Make the initial call to list secrets
            response = self.secrets_client.list_secrets(**params)

            # Filter and aggregate the names from the initial response
            for secret in response["SecretList"]:
                name = secret["Name"]
                if not prefix or name.startswith(prefix):
                    all_secrets.append(name)

            # Continue to paginate if there's a NextToken
            while "NextToken" in response:
                # Update the parameters with the NextToken for subsequent calls
                params["NextToken"] = response["NextToken"]
                response = self.secrets_client.list_secrets(**params)

                # Filter and aggregate the names from the subsequent responses
                for secret in response["SecretList"]:
                    name = secret["Name"]
                    if not prefix or name.startswith(prefix):
                        all_secrets.append(name)

        except Exception as e:
            self.log.error(f"Failed to list secrets: {e}")
            return []

        # Return the aggregated list of secret names
        return all_secrets

    def get(self, name: str, warn: bool = True) -> Union[str, list, dict, None]:
        """Retrieve a secret value from AWS Secrets Manager.

        Args:
            name (str): The name of the secret to retrieve.
            warn (bool): Whether to log a warning if the secret is not found.

        Returns:
            Union[str, list, dict, None]: The value of the secret or None if not found.
        """
        try:
            # Retrieve the secret from Secrets Manager
            response = self.secrets_client.get_secret_value(SecretId=name)

            # Get the secret value (could be in SecretString or SecretBinary)
            if "SecretString" in response:
                value = response["SecretString"]
            elif "SecretBinary" in response:
                value = response["SecretBinary"]
            else:
                self.log.warning(f"Secret '{name}' doesn't contain SecretString or SecretBinary")
                return None

            # Attempt to parse the value back to its original type
            try:
                parsed_value = json.loads(value)
                return parsed_value
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, return the value as is (assumed to be a simple string)
                return value

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                if warn:
                    self.log.warning(f"Secret '{name}' not found")
            else:
                self.log.error(f"Failed to get secret '{name}': {e}")
            return None

    def upsert(self, name: str, value: Union[str, list, dict]) -> bool:
        """Create or update a secret in AWS Secrets Manager.

        Args:
            name (str): The name of the secret to create or update.
            value (Union[str, list, dict]): The value to store.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Convert complex types to JSON strings
            if not isinstance(value, str):
                value = json.dumps(value)

            # Check if the secret already exists
            try:
                self.secrets_client.describe_secret(SecretId=name)
                # Secret exists, update it
                self.secrets_client.update_secret(SecretId=name, SecretString=value)
                self.log.info(f"Updated secret '{name}'")
            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceNotFoundException":
                    # Secret doesn't exist, create it
                    self.secrets_client.create_secret(Name=name, SecretString=value)
                    self.log.info(f"Created secret '{name}'")
                else:
                    raise e

            return True

        except Exception as e:
            self.log.error(f"Failed to upsert secret '{name}': {e}")
            return False

    def delete(self, name: str, force=False) -> bool:
        """Delete a secret from AWS Secrets Manager.

        Args:
            name (str): The name of the secret to delete.
            force (bool): Whether to force delete the secret without recovery.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if force:
                # Force delete the secret without recovery
                self.secrets_client.delete_secret(SecretId=name, ForceDeleteWithoutRecovery=True)
                self.log.info(f"Force deleted secret '{name}' (not recoverable)")
                return True
            # Delete the secret with recovery (30 days)
            self.secrets_client.delete_secret(SecretId=name, ForceDeleteWithoutRecovery=False)
            self.log.info(f"Deleted secret '{name}' (recoverable for 30 days)")
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                self.log.warning(f"Secret '{name}' not found for deletion")
                return True  # Consider it a success if it didn't exist anyway
            else:
                self.log.error(f"Failed to delete secret '{name}': {e}")
                return False


if __name__ == "__main__":
    """Exercise the AWS Secrets Manager class"""
    # Create an instance of the AWSSecretsManager
    secrets_manager = AWSSecretsManager()

    # List all secrets
    print("Listing all secrets:")
    print(secrets_manager.list())

    # Add a secret
    secrets_manager.upsert("test_api_key", "my-secret-key")
    print("Added secret 'test_api_key'")

    # Retrieve the secret
    api_key = secrets_manager.get("test_api_key")
    print(f"Retrieved secret 'test_api_key': {api_key}")

    # Add structured data
    credentials = {"username": "admin", "password": "secret123", "host": "db.example.com"}
    secrets_manager.upsert("test_database_credentials", credentials)
    print("Added secret 'test_database_credentials'")

    # Retrieve the structured data
    db_creds = secrets_manager.get("test_database_credentials")
    print(f"Retrieved secret 'test_database_credentials': {db_creds}")

    # Delete the secret
    secrets_manager.delete("test_database_credentials", force=True)
    print("Deleted secret 'test_database_credentials'")
