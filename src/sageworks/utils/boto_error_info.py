import botocore
import logging

# SageWorks Logger
log = logging.getLogger("sageworks")


def client_error_info(err: botocore.exceptions.ClientError):
    """Helper method to get information about a botocore.exceptions.ClientError"""
    error_code = err.response["Error"]["Code"]
    error_message = err.response["Error"]["Message"]
    operation_name = err.operation_name
    service_name = err.response["ResponseMetadata"].get("ServiceName", "Unknown")
    request_id = err.response["ResponseMetadata"]["RequestId"]

    # Output the error information
    log.error(f"Error Code: {error_code}")
    log.error(f"Error Message: {error_message}")
    log.error(f"Operation Name: {operation_name}")
    log.error(f"Service Name: {service_name}")
    log.error(f"Request ID: {request_id}")
