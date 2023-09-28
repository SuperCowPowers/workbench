import json
import os
import boto3


def lambda_handler(event, context):
    # Environment vars
    DOCKER_REPO = os.environ["DOCKER_REPO"]
    TASK_DEFINITION = os.environ["TASK_DEFINITION"]
    FARGATE_CLUSTER = os.environ["FARGATE_CLUSTER"]
    FARGATE_SERVICE = os.environ["FARGATE_SERVICE"]

    # Pull docker image from event
    docker_image_tag = event["detail"]["image-tag"][0]
    docker_image = f"{DOCKER_REPO}:{docker_image_tag}"

    # Connect to client
    ecs = boto3.client("ecs")

    # Maybe pull task definition from existing service and copy over?
    old_task_def = ecs.describe_task_definition(taskDefinition=TASK_DEFINITION)

    # Change image to new image in task definition
    new_task_def = old_task_def["taskDefinition"]
    new_task_def["containerDefinitions"][0]["image"] = docker_image

    # Drop keys that can't be used as kwargs
    remove_args = [
        "compatibilities",
        "registeredAt",
        "registeredBy",
        "status",
        "revision",
        "taskDefinitionArn",
        "requiresAttributes",
    ]
    for arg in remove_args:
        new_task_def.pop(arg)

    # Register new task definition
    new_task_res = ecs.register_task_definition(**new_task_def)
    new_task_def_arn = new_task_res["taskDefinition"]["taskDefinitionArn"]

    # Define new deployment strategy
    deploymentConfiguration = {"maximumPercent": 200, "minimumHealthyPercent": 100}

    # Update service with new task definition
    ecs.update_service(
        cluster=FARGATE_CLUSTER,
        service=FARGATE_SERVICE,
        taskDefinition=new_task_def_arn,
        deploymentConfiguration=deploymentConfiguration,
    )

    res = {"newDockerImageTag": docker_image_tag, "newTaskDefArn": new_task_def_arn}
    return {"statusCode": 200, "body": json.dumps(res)}
