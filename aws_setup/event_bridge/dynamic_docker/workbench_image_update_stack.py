# CDK Imports
import aws_cdk as cdk
from constructs import Construct
from aws_cdk import aws_iam as iam
from aws_cdk.aws_lambda import Function, Runtime, Code
from aws_cdk.aws_events import Rule, EventPattern
from aws_cdk import aws_events_targets as targets

# Other imports
import os

# TODO Globals (most of these variables will need to be pulled from the main dashboard stack)
DOCKER_REPO = "public.ecr.aws/m6i5k1r2/workbench_dashboard"
TASK_DEFINITION = "arn:aws:ecs:us-east-1:116553368710:task-definition/WorkbenchDashboardWorkbenchTaskDefDC75FE98"
FARGATE_SERVICE = "arn:aws:ecs:us-east-1:116553368710:service/WorkbenchDashboard-WorkbenchCluster50ABAB0B-4UgZDrID6sRK/WorkbenchDashboard-WorkbenchService23524EF1-6BePIi17bxPj"
FARGATE_CLUSTER = "arn:aws:ecs:us-east-1:116553368710:cluster/WorkbenchDashboard-WorkbenchCluster50ABAB0B-4UgZDrID6sRK"


class WorkbenchImageUpdateStack(cdk.Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Import the existing Workbench-ExecutionRole
        workbench_execution_role = iam.Role.from_role_arn(
            self, "ImportedWorkbenchExecutionRole", f"arn:aws:iam::{self.account}:role/Workbench-ExecutionRole"
        )

        # Instantiate lambda function
        lambda_func = Function(
            self,
            "Update-workbench-dashboard-task-image",
            runtime=Runtime.PYTHON_3_11,
            handler="lambda_replace_task.lambda_handler",
            code=Code.from_asset(os.path.join(os.getcwd(), "workbench_image_update", "lambda")),
            environment={
                "DOCKER_REPO": DOCKER_REPO,
                "TASK_DEFINITION": TASK_DEFINITION,
                "FARGATE_CLUSTER": FARGATE_CLUSTER,
                "FARGATE_SERVICE": FARGATE_SERVICE,
            },
            role=workbench_execution_role,
            timeout=cdk.Duration.seconds(300),
        )

        # Instantiate EventBridge Rule
        # At this point this only works with ECRs in the same account, probably because it's a push rather than pull mechanism
        rule = Rule(
            self,
            "Detect-new-workbench-image",
            rule_name="Detect-new-workbench-image",
            event_pattern=EventPattern(
                source=["aws.ecr"],
                detail_type=["ECR Image Action"],
                detail={"action-type": ["PUSH"], "result": ["SUCCESS"], "repository-name": [DOCKER_REPO]},
            ),
        )

        # Add Lambda as target
        rule.add_target(targets.LambdaFunction(lambda_func))
