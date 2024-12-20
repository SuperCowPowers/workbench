import json
import logging
from pprint import pprint
from botocore.exceptions import ClientError

# Local Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp


class WorkbenchEventBridge:
    def __init__(self, bus_name="workbench"):
        """WorkbenchEventBridge: Class for publishing logging events to AWS EventBridge"""
        self.log = logging.getLogger("workbench")
        self.event_bus = bus_name
        self.event_bridge = None

        # Grab a Workbench Session (this allows us to assume the Workbench ExecutionRole)
        self.boto3_session = AWSAccountClamp().boto3_session

        # Check if the EventBridge bus exists
        self._check_event_bus()

    def _check_event_bus(self):
        """Check if the event bus exists and set up the EventBridge client"""
        try:
            # Get our AWS EventBridge Client
            event_bridge_client = self.boto3_session.client("events")

            # Describe the event bus to check if it exists
            event_bridge_client.describe_event_bus(Name=self.event_bus)

            # If no exception, set the event_bridge client for future use
            self.event_bridge = event_bridge_client
            self.log.info(f"Connected to EventBridge event bus: {self.event_bus}")

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                self.log.error(f"Event bus '{self.event_bus}' does not exist. " "Event messages will not be sent.")
            else:
                self.log.error(f"Failed to connect to EventBridge: {e}")
            # EventBridge client remains None, and events won't be sent

    def event_warning(self, message: str, **kwargs):
        event = {
            "eventType": "WARNING",
            "message": message,
            "details": kwargs,
        }
        return self.send_event(event, "LoggingEvent")

    def event_error(self, message: str, **kwargs):
        event = {
            "eventType": "ERROR",
            "message": message,
            "details": kwargs,
        }
        return self.send_event(event, "LoggingEvent")

    def event_critical(self, message: str, **kwargs):
        event = {
            "eventType": "CRITICAL",
            "message": message,
            "details": kwargs,
        }
        return self.send_event(event, "LoggingEvent")

    def send_event(self, event, detail_type):
        # If the event_bridge is not set, log a warning and return without sending
        if self.event_bridge is None:
            self.log.warning("Event not sent. EventBridge bus is unavailable.")
            return None

        event_data = {
            "Detail": json.dumps(event),
            "DetailType": detail_type,
            "Source": "com.supercowpowers.workbench",
            "EventBusName": self.event_bus,
        }
        response = self.event_bridge.put_events(Entries=[event_data])

        # Check the status code of the response
        status = response["ResponseMetadata"]["HTTPStatusCode"]
        if status != 200:
            self.log.critical(f"Event Bridge Failure: {response}")
            return None

        # Check that we got back an EventID
        entry_status = response["Entries"][0]
        if entry_status.get("EventId") is None:
            code = entry_status["ErrorCode"]
            message = entry_status["ErrorMessage"]
            self.log.critical(f"Event Bridge Failure: {code} {message}")
            return None

        # AOK so return the response in case the caller wants it
        return response


def test():
    """Run a test for WorkbenchEventBridge class"""

    event_bridge = WorkbenchEventBridge()

    # Send some test EventBridge messages
    response = event_bridge.event_warning("This is a test warning", component="TestComponent")
    pprint(response)
    response = event_bridge.event_error("This is a test error", component="TestComponent")
    pprint(response)
    response = event_bridge.event_critical("This is a test critical error", component="TestComponent")
    pprint(response)


if __name__ == "__main__":
    # Run the test
    test()
