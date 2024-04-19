"""PipelineExecutor: Internal Class: Executes a SageWorks Pipeline"""

import logging

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet, Model, Endpoint
from sageworks.api.model import ModelType


class PipelineExecutor:
    """PipelineExecutor: Internal Class: Executes a SageWorks Pipeline

    Common Usage:
        ```
        my_pipeline = PipelineExecutor(pipeline)
        my_pipeline.execute()  # Execute entire pipeline
        my_pipeline.execute_partial(["data_source", "feature_set"])
        my_pipeline.execute_partial(["model", "endpoint"])
        ```
    """

    def __init__(self, pipeline):
        """PipelineExecutor Init Method"""
        self.log = logging.getLogger("sageworks")
        self.pipeline_name = pipeline.pipeline_name
        self.pipeline = pipeline.pipeline

    def execute(self, subset: list = None):
        """Execute the SageWorks Pipeline

        Args:
            subset (list): A list of steps to execute. If None, execute the entire pipeline

        Raises:
            RuntimeError: If the pipeline execution fails in any way
        """
        self.log.important(f"Executing Pipeline {self.pipeline_name}...")
        if subset:
            self.log.important(f"\tSubset: {subset}")
        sageworks_objects = {}
        for class_name, kwargs in self.pipeline.items():

            # Input is a special case
            input = kwargs["input"]
            del kwargs["input"]
            if isinstance(input, str) and input == "DataFrame":
                msg = "Call set_input() to set the input DataFrame"
                self.log.critical(msg)
                raise RuntimeError(msg)

            # DataSource
            if class_name == "data_source":
                # Create a DataSource only when it's implicit or explicitly requested
                if not subset or "data_source" in subset:
                    sageworks_objects["data_source"] = DataSource(input, **kwargs)

            # FeatureSet
            elif class_name == "feature_set":

                # Special case for hold_out_ids
                if "hold_out_ids" in kwargs:
                    if kwargs["hold_out_ids"] == "<<dynamic>>":
                        self.log.warning("Hold out ids are dynamic and not set, defaulting to 80/20 split")

                # Check for a transform and create a FeatureSet
                if "data_source" in sageworks_objects and not subset or "feature_set" in subset:
                    sageworks_objects["data_source"].to_features(**kwargs)
                if not subset or "model" in subset:
                    sageworks_objects["feature_set"] = FeatureSet(kwargs["name"])

            # Model
            elif class_name == "model":

                # Special case for model type
                if "model_type" in kwargs:
                    kwargs["model_type"] = ModelType(kwargs["model_type"])

                # Check for a transform and create a Model
                if "feature_set" in sageworks_objects:
                    sageworks_objects["feature_set"].to_model(**kwargs)
                if not subset or "endpoint" in subset:
                    sageworks_objects["model"] = Model(kwargs["name"])
                    sageworks_objects["model"].set_owner("pipeline")

            # Endpoint
            elif class_name == "endpoint":
                # Check for a transform
                if "model" in sageworks_objects and not subset or "endpoint" in subset:
                    sageworks_objects["model"].to_endpoint(**kwargs)
                    endpoint = Endpoint(kwargs["name"])
                    endpoint.auto_inference(capture=True)

            # Found something weird
            else:
                raise RuntimeError(f"Unsupported pipeline stage: {class_name}")

    def execute_partial(self, subset: list):
        """Execute a partial Pipeline

        Args:
            subset (list): A subset of the pipeline to execute

        Raises:
            RunTimeException: If the pipeline execution fails in any way
        """
        self.execute(subset)


if __name__ == "__main__":
    """Exercise the PipelineExecutor Class"""
    from sageworks.api.pipeline import Pipeline

    # Retrieve an existing Pipeline
    pipeline = Pipeline("abalone_pipeline_v1")

    pipeline_executor = PipelineExecutor(pipeline)

    # Execute the PipelineExecutor
    # pipeline_executor.execute()

    # Execute partial Pipelines
    # pipeline_executor.execute_partial(["data_source"])
    # pipeline_executor.execute_partial(["data_source", "feature_set"])
    # pipeline_executor.execute_partial(["model", "endpoint"])
