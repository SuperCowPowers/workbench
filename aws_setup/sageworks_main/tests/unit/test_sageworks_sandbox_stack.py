import aws_cdk as core
from aws_cdk.assertions import Template

from sageworks_sandbox.sageworks_sandbox_stack import SageworksStack


# example tests. To run these tests, uncomment this file along with the example
# resource in sageworks_sandbox/sageworks_sandbox_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = SageworksStack(
        app,
        "sageworks-sandbox",
        "test-scp-sageworks-artifacts",
        "SageWorks-ExecutionRole",
    )
    template = Template.from_stack(stack)

    # TODO: Check for specific properties
    template.resource_count_is("AWS::IAM::Role", 1)
    template.resource_count_is("AWS::IAM::Policy", 1)
    template.resource_count_is("AWS::S3::Bucket", 1)
