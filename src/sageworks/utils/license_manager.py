"""Internal: SageWorks API License Manager (used by ConfigManager, do not use directly)"""

import sys
import base64
import json
import logging
import requests
from typing import Union
import importlib.resources as resources
from datetime import datetime
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend


class FatalLicenseError(Exception):
    """Exception raised for fatal errors in API License."""

    def __init__(self):
        sys.exit(1)


class LicenseManager:
    """Internal: SageWorks API License Manager (used by ConfigManager, do not use directly)"""

    api_license_info = None
    log = logging.getLogger("sageworks")

    @classmethod
    def load_api_license(
        cls, aws_account_id: Union[str, None], api_key: str, license_api_key: str = None
    ) -> Union[dict, None]:
        """Internal: Load the SageWorks API License, verify it, and return the licensed features
        Args:
            aws_account_id(str): The AWS Account ID to verify the license against (None for Open Source)
            api_key(str): The SageWorks API Key to verify (base64 encoded)
            license_api_key(str): The SageWorks License API Key (default: None)
        Returns:
            dict/None: The SageWorks API License Information or None if the license is invalid
        """

        # Store the API Key for later use
        cls.api_key = api_key
        cls.license_api_key = license_api_key

        # Decode the API Key
        try:
            decoded_license_key = base64.urlsafe_b64decode(api_key)
            _license_data, signature = cls.extract_data_and_signature(decoded_license_key)
        except Exception as e:
            cls.log.critical(f"Failed to decode API Key: {e}")
            cls.log.critical("Please contact SageWorks support")
            raise FatalLicenseError()

        # Verify the signature of the API Key
        if not cls.verify_signature(_license_data, signature):
            msg = "API License key verification failed, Please contact SageWorks support"
            cls.log.critical(msg)
            raise FatalLicenseError()

        # Load the license data into a dictionary
        cls.api_license_info = json.loads(_license_data)

        # Check if the API license is expired
        if cls.is_license_expired():
            cls.log.critical(
                f"API License expired on {cls.api_license_info.get('expires')} Please contact SageWorks support."
            )
            raise FatalLicenseError()

        # Grab the AWS Account ID from our API License
        api_account_id = cls.api_license_info.get("aws_account_id")

        # Check if the API License is valid for this AWS Account
        if api_account_id and aws_account_id and api_account_id != aws_account_id:
            cls.log.critical("SageWorks API Key is not valid for this AWS Account!")
            cls.log.critical(f"Connected AWS Account ID: {aws_account_id}")
            cls.log.critical(f"API License AWS Account ID: {api_account_id}")
            raise FatalLicenseError()

        # Return the license information
        return cls.api_license_info

    @classmethod
    def print_license_info(cls):
        id = cls.api_license_info["license_id"]
        account = cls.api_license_info["aws_account_id"]
        expires = cls.api_license_info["expires"]
        cls.log.important(f"SageWorks License: {id}-{account}-{expires}")

    @staticmethod
    def extract_data_and_signature(license_key):
        # Data is everything except the last 128 bytes
        license_data = license_key[:-128].decode("utf-8")
        signature = license_key[-128:]
        return license_data, signature

    @classmethod
    def verify_signature(cls, license_data, signature):
        public_key = cls.read_signature_public_key()
        try:
            public_key.verify(
                signature,
                license_data.encode("utf-8"),  # Encode license data as bytes
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )
            return True
        except Exception as e:
            print(f"Signature verification failed: {e}")
            return False

    @classmethod
    def is_license_expired(cls):
        expiration_date = cls.api_license_info.get("expires")
        if not expiration_date:
            return True

        # Convert expiration_date string to datetime object
        expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d")
        return datetime.now() > expiration_date

    @classmethod
    def get_license_id(cls) -> str:
        """Get the license ID from the license information
        Returns:
            str: The license ID
        """
        return cls.api_license_info.get("license_id", "Unknown") if cls.api_license_info else "Unknown"

    @staticmethod
    def read_signature_public_key():
        """Read the public key from the package.
        Returns:
            The public key as an object.
        """
        with resources.path("sageworks.resources", "signature_verify_pub.pem") as public_key_path:
            with open(public_key_path, "rb") as key_file:
                public_key_data = key_file.read()

        public_key = serialization.load_pem_public_key(public_key_data, backend=default_backend())
        return public_key

    @classmethod
    def contact_license_server(cls) -> requests.Response:
        """Contact the SageWorks License Server to verify the license."""
        server_url = "https://sageworks-keyserver.com/decode-key"
        headers = {"Content-Type": "application/json", "x-api-key": cls.license_api_key}
        data = {"api_key": cls.api_key}
        return requests.post(server_url, headers=headers, json=data)


if __name__ == "__main__":
    """Exercise the License Manager class"""
    from sageworks.utils.config_manager import ConfigManager

    # Grab the API Key from the SageWorks ConfigManager
    cm = ConfigManager()
    api_key = cm.get_config("SAGEWORKS_API_KEY")
    license_api_key = cm.get_config("LICENSE_API_KEY")
    print(LicenseManager.get_license_id())

    my_license_info = LicenseManager.load_api_license(
        aws_account_id=None, api_key=api_key, license_api_key=license_api_key
    )
    print(my_license_info)
    LicenseManager.print_license_info()
    print(LicenseManager.get_license_id())

    # Test the license server
    response = LicenseManager.contact_license_server()
    print(response.json())
