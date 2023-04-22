import aws_cdk as core
import aws_cdk.assertions as assertions

from sageworks_sandbox.sageworks_sandbox_stack import SageworksSandboxStack

# example tests. To run these tests, uncomment this file along with the example
# resource in sageworks_sandbox/sageworks_sandbox_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = SageworksSandboxStack(app, "sageworks-sandbox")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
