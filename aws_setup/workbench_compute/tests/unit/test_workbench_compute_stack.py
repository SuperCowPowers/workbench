import aws_cdk as core
import aws_cdk.assertions as assertions

from workbench_compute.workbench_compute_stack import WorkbenchComputeStack


# example tests. To run these tests, uncomment this file along with the example
# resource in workbench_compute/workbench_compute_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = WorkbenchComputeStack(app, "workbench-compute")
    template = assertions.Template.from_stack(stack)


#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
