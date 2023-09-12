import aws_cdk as cdk

from stacks.sageworks_stack import SageworksStack
from sageworks.utils.sageworks_config import SageWorksConfig

app = cdk.App()

sageworks_config: SageWorksConfig = SageWorksConfig()
s3_bucket_name = sageworks_config.get_config_value("SAGEWORKS_AWS", "S3_BUCKET_NAME")
sageworks_role_name = sageworks_config.get_config_value("SAGEWORKS_AWS", "SAGEWORKS_ROLE_NAME")

sandbox_stack = SageworksStack(
    app,
    "Sageworks",
    s3_bucket_name=s3_bucket_name,
    sageworks_role_name=sageworks_role_name
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
