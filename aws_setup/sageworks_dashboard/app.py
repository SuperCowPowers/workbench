import os

import aws_cdk as cdk

from sageworks_dashboard.sageworks_dashboard_stack import SageworksDashboardStack

# We'd like to set up our parameters here and NOT have them handled in the stack
# Example: VPC, Subnet, Security Group, etc.
# Perhaps a config file instead of ENV vars?
app = cdk.App()
SageworksDashboardStack(
    app,
    "SageworksDashboard",
    # If you don't specify 'env', this stack will be environment-agnostic.
    # Account/Region-dependent features and context lookups will not work,
    # but a single synthesized template can be deployed anywhere.
    # Uncomment the next line to specialize this stack for the AWS Account
    # and Region that are implied by the current CLI configuration.
    # env=cdk.Environment(account=os.getenv('CDK_DEFAULT_ACCOUNT'), region=os.getenv('CDK_DEFAULT_REGION')),
    # Uncomment the next line if you know exactly what Account and Region you
    # want to deploy the stack to. */
    # env=cdk.Environment(account='123456789012', region='us-east-1'),
    # For more information, see https://docs.aws.amazon.com/cdk/latest/guide/environments.html
)

app.synth()
