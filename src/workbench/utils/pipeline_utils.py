"""Workbench Pipeline Utilities"""

import logging
import json

# Workbench Imports
from workbench.api import DataSource, FeatureSet, Model, Endpoint, ParameterStore


# Set up the logging
log = logging.getLogger("workbench")


# Create a new Pipeline from an Endpoint
def create_from_endpoint(endpoint_name: str) -> dict:
    """Create a Pipeline from an Endpoint

    Args:
        endpoint_name (str): The name of the Endpoint

    Returns:
        dict: A dictionary of the Pipeline
    """
    log.important(f"Creating Pipeline from Endpoint: {endpoint_name}...")
    pipeline = {}
    endpoint = Endpoint(endpoint_name)
    model = Model(endpoint.get_input())
    feature_set = FeatureSet(model.get_input())
    data_source = DataSource(feature_set.get_input())
    s3_source = data_source.get_input()
    for name in ["data_source", "feature_set", "model", "endpoint"]:
        artifact = locals()[name]
        pipeline[name] = {"name": artifact.uuid, "tags": artifact.get_tags(), "input": artifact.get_input()}
        if name == "model":
            pipeline[name]["model_type"] = artifact.model_type.value
            pipeline[name]["target_column"] = artifact.target()
            pipeline[name]["feature_list"] = artifact.features()

    # Return the Pipeline
    return pipeline


def publish_pipeline(name: str, pipeline: dict):
    """Publish a Pipeline to Parameter Store

    Args:
        name (str): The name of the Pipeline
        pipeline (dict): The Pipeline to save
    """
    params = ParameterStore()
    key = f"/workbench/pipelines/{name}"
    log.important(f"Saving {name} to Parameter Store {key}...")

    # Save the pipeline to the parameter store
    params.upsert(key, json.dumps(pipeline))


if __name__ == "__main__":
    """Exercise the Pipeline Class"""
    from pprint import pprint
    from workbench.api.meta import Meta

    # List the Pipelines
    meta = Meta()
    print("Listing Pipelines...")
    pprint(meta.pipelines())

    # Create a Pipeline from an Endpoint
    abalone_pipeline = create_from_endpoint("abalone-regression")

    # Publish the Pipeline
    publish_pipeline("abalone_pipeline_test", abalone_pipeline)
