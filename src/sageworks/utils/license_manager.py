"""SageWorks API License Manager"""
import os
import sys
import base64
import json
import logging
from typing import Union
import pkg_resources
from datetime import datetime
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

import sageworks  # noqa: F401 (we need to import this to set up the logger)


class LicenseManager:
    API_ENV_VAR = "SAGEWORKS_API_KEY"
    api_license_info = None
    log = logging.getLogger("sageworks")

    @classmethod
    def load_api_license(cls, aws_account_id: str) -> Union[dict, None]:
        """Load the SageWorks API License, verify it, and return the licensed features
        Args:
            aws_account_id(str): The AWS Account ID to verify the license against
        Returns:
            dict/None: The SageWorks API License Information or None if the license is invalid
        """

        api_key = os.getenv(cls.API_ENV_VAR)
        if not api_key:
            cls.log.important(f"Could not find ENV var for {cls.API_ENV_VAR}, using Open Source API Key...")
            cls.set_open_source_api_key()
            api_key = os.getenv(cls.API_ENV_VAR)
            if not api_key:
                cls.log.critical("Could not set Open Source API Key!")
                return None

        # Decode the API Key
        decoded_license_key = base64.b64decode(api_key)
        _license_data, signature = cls.extract_data_and_signature(decoded_license_key)

        # Verify the signature of the API Key
        if not cls.verify_signature(_license_data, signature):
            msg = "API License key verification failed."
            cls.log.critical(msg)
            return None

        # Load the license data into a dictionary
        cls.api_license_info = json.loads(_license_data)

        # Check if the API license is expired
        if cls.is_license_expired():
            cls.log.critical(
                f"API License expired on {cls.api_license_info.get('expires')} Please contact SageWorks support."
            )
            return None

        # Verify our AWS Account ID
        api_account_id = cls.api_license_info.get("aws_account_id")

        # Open Source License has no AWS Account ID
        if api_account_id is None:
            return cls.api_license_info

        # Check if the API License is valid for this AWS Account
        if api_account_id != aws_account_id:
            cls.log.critical("SageWorks API Key is not valid for this AWS Account!")
            cls.log.critical(f"Connected AWS Account ID: {aws_account_id}")
            cls.log.critical(f"API License AWS Account ID: {api_account_id}")
            sys.exit(1)

        # Return the license information
        return cls.api_license_info

    @classmethod
    def print_license_info(cls):
        cls.log.important("License Info:")
        cls.log.important(json.dumps(cls.api_license_info, indent=4, sort_keys=True))

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
        return cls.api_license_info.get("license_id", "Unknown")

    @staticmethod
    def read_signature_public_key():
        """Read the public key from the package.
        Returns:
            The public key as an object.
        """
        public_key_path = pkg_resources.resource_filename("sageworks", "resources/signature_verify_pub.pem")
        with open(public_key_path, "rb") as key_file:
            public_key_data = key_file.read()

        public_key = serialization.load_pem_public_key(public_key_data, backend=default_backend())
        return public_key

    @classmethod
    def set_open_source_api_key(cls):
        """Read the public key from the package.
        Returns:
            The public key as an object.
        """
        open_source_key_path = pkg_resources.resource_filename("sageworks", "resources/open_source_api.key")
        with open(open_source_key_path, "rb") as key_file:
            open_source_key = key_file.read().decode("utf-8").strip()
        os.environ[cls.API_ENV_VAR] = open_source_key


if __name__ == "__main__":
    """Exercise the License Manager class"""

    my_license_info = LicenseManager.load_api_license(aws_account_id="123")
    print(my_license_info)
    LicenseManager.print_license_info()
    print(LicenseManager.get_license_id())
