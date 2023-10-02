"""SageWorksEventBridge: Class for publishing events to AWS EventBridge"""
import json
from pprint import pprint
import logging

# Local Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.transforms.transform import TransformOutput as ArtifactType
from sageworks.utils.sageworks_logging import logging_setup

# Setup Logging
logging_setup()


class SageWorksEventBridge:
    def __init__(self, bus_name="sageworks"):
        """SageWorksEventBridge: Class for publishing events to AWS EventBridge"""
        self.log = logging.getLogger("sageworks")
        self.event_bus = bus_name

        # Grab a SageWorks Session (this allows us to assume the SageWorks ExecutionRole)
        self.boto_session = AWSAccountClamp().boto_session()

        # Get our AWS EventBridge Client
        self.event_bridge = self.boto_session.client("events")

    def create_artifact(self, uuid: str, artifact_type: ArtifactType):
        event = {
            "eventType": "CREATE_ARTIFACT",
            "artifactType": artifact_type.name,
            "uuid": uuid,
        }
        return self.send_event(event, "ArtifactChange")

    def modify_artifact(self, uuid: str, artifact_type: ArtifactType):
        event = {
            "eventType": "MODIFY_ARTIFACT",
            "artifactType": artifact_type.name,
            "uuid": uuid,
        }
        return self.send_event(event, "ArtifactChange")

    def archive_artifact(self, uuid: str, artifact_type: ArtifactType):
        event = {
            "eventType": "ARCHIVE_ARTIFACT",
            "artifactType": artifact_type.name,
            "uuid": uuid,
        }
        return self.send_event(event, "ArtifactChange")

    def delete_artifact(self, uuid: str, artifact_type: ArtifactType):
        event = {
            "eventType": "DELETE_ARTIFACT",
            "artifactType": artifact_type.name,
            "uuid": uuid,
        }
        return self.send_event(event, "ArtifactChange")

    def send_event(self, event, detail_type):
        event_data = {
            "Detail": json.dumps(event),
            "DetailType": detail_type,
            "Source": "com.supercowpowers.sageworks",
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
    """Run a test for EventBridge class"""

    event_bridge = SageWorksEventBridge()

    # Send some test EventBridge messages
    response = event_bridge.create_artifact("123", ArtifactType.DATA_SOURCE)
    pprint(response)
    response = event_bridge.modify_artifact("123", ArtifactType.DATA_SOURCE)
    pprint(response)
    response = event_bridge.archive_artifact("123", ArtifactType.DATA_SOURCE)
    pprint(response)
    response = event_bridge.delete_artifact("123", ArtifactType.DATA_SOURCE)
    pprint(response)


if __name__ == "__main__":
    # Run the test
    test()
