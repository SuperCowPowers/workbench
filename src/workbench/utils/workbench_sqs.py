"""WorkbenchSQS: Class for retrieving messages from the AWS SQS Message Queue"""

import logging

# Local Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp


class WorkbenchSQS:
    def __init__(self, queue_url="workbench.fifo"):
        """WorkbenchSQS: Class for retrieving messages from the AWS SQS Message Queue"""
        self.log = logging.getLogger("workbench")
        self.queue_url = queue_url

        # Grab a Workbench Session (this allows us to assume the Workbench-ExecutionRole)
        self.boto3_session = AWSAccountClamp().boto3_session
        print(self.boto3_session)

        # Get our AWS EventBridge Client
        self.sqs = self.boto3_session.client("sqs")

    def get_message(self, delete=False):
        """Get a message from the SQS Message Queue
        Args:
            delete (bool): Delete the message from the Queue after reading it
        """
        response = self.sqs.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=1,
            MessageAttributeNames=["All"],
            VisibilityTimeout=0,
            WaitTimeSeconds=0,
        )

        # Check the status code of the response
        status = response["ResponseMetadata"]["HTTPStatusCode"]
        if status != 200:
            self.log.critical(f"SQS Get Message Failure: {response}")
            return None

        # Grab the message and receipt handle
        message = response["Messages"][0]
        receipt_handle = message["ReceiptHandle"]
        message_body = message["Body"]

        # Delete received message from queue
        if delete:
            self.sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)

        # Return the message body
        return message_body


def test():
    """Run a test for EventBridge class"""

    # Create the Class and Get some messages
    workbench_sqs = WorkbenchSQS()
    message = workbench_sqs.get_message(delete=False)
    print(message)
    message = workbench_sqs.get_message(delete=False)
    print(message)


if __name__ == "__main__":
    # Run the test
    test()
