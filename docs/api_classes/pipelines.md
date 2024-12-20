# Pipelines

!!! tip inline end "Pipeline Examples"
    Examples of using the Pipeline classes are listed at the bottom of this page [Examples](#examples).
    
Pipelines store sequences of Workbench transforms. So if you have a nightly ML workflow you can capture that as a Pipeline. Here's an example pipeline:

```py title="nightly_sol_pipeline_v1.json"
{
    "data_source": {
         "name": "nightly_data",
         "tags": ["solubility", "foo"],
         "s3_input": "s3://blah/blah.csv"
    },
    "feature_set": {
          "name": "nightly_features",
          "tags": ["blah", "blah"],
          "input": "nightly_data"
          "schema": "mol_descriptors_v1"
    },
    "model": {
          "name": “nightly_model”,
          "tags": ["blah", "blah"],
          "features": ["col1", "col2"],
          "target": “sol”,
          "input": “nightly_features”
    "endpoint": {
          ...
}    
```

::: workbench.api.pipeline


## Examples

**Make a Pipeline**

Pipelines are just JSON files (see `workbench/examples/pipelines/`). You can copy one and make changes to fit your objects/use case, or if you have a set of Workbench artifacts created you can 'backtrack' from the Endpoint and have it create the Pipeline for you.

```py title="pipeline_manager.py"
from workbench.api.pipeline_manager import PipelineManager

 # Create a PipelineManager
my_manager = PipelineManager()

# List the Pipelines
pprint(my_manager.list_pipelines())

# Create a Pipeline from an Endpoint
abalone_pipeline = my_manager.create_from_endpoint("abalone-regression-end")

# Publish the Pipeline
my_manager.publish_pipeline("abalone_pipeline_v1", abalone_pipeline)
```

**Output**

```py
Listing Pipelines...
[{'last_modified': datetime.datetime(2024, 4, 16, 21, 10, 6, tzinfo=tzutc()),
  'name': 'abalone_pipeline_v1',
  'size': 445}]
```
**Pipeline Details**

```py title="pipeline_details.py"
from workbench.api.pipeline import Pipeline

# Retrieve an existing Pipeline
my_pipeline = Pipeline("abalone_pipeline_v1")
pprint(my_pipeline.details())
```

**Output**

```json
{
    "name": "abalone_pipeline_v1",
    "s3_path": "s3://sandbox/pipelines/abalone_pipeline_v1.json",
    "pipeline": {
        "data_source": {
            "name": "abalone_data",
            "tags": [
                "abalone_data"
            ],
            "input": "/Users/briford/work/workbench/data/abalone.csv"
        },
        "feature_set": {
            "name": "abalone_features",
            "tags": [
                "abalone_features"
            ],
            "input": "abalone_data"
        },
        "model": {
            "name": "abalone-regression",
            "tags": [
                "abalone",
                "regression"
            ],
            "input": "abalone_features"
        },
        ...
    }
}
```

**Pipeline Execution**

!!!tip inline end "Pipeline Execution" 
    Executing the Pipeline is obviously the most important reason for creating one. If gives you a reproducible way to capture, inspect, and run the same ML pipeline on different data (nightly).

```py title="pipeline_execution.py"
from workbench.api.pipeline import Pipeline

# Retrieve an existing Pipeline
my_pipeline = Pipeline("abalone_pipeline_v1")

# Execute the Pipeline
my_pipeline.execute()  # Full execution

# Partial executions
my_pipeline.execute_partial(["data_source", "feature_set"])
my_pipeline.execute_partial(["model", "endpoint"])
```

## Pipelines Advanced
As part of the flexible architecture sometimes DataSources or FeatureSets can be created with a Pandas DataFrame. To support a DataFrame as input to a pipeline we can call the `set_input()` method to the pipeline object. If you'd like to specify the `set_hold_out_ids()` you can also provide a list of ids.

```
    def set_input(self, input: Union[str, pd.DataFrame], artifact: str = "data_source"):
        """Set the input for the Pipeline

        Args:
            input (Union[str, pd.DataFrame]): The input for the Pipeline
            artifact (str): The artifact to set the input for (default: "data_source")
        """
        self.pipeline[artifact]["input"] = input

    def set_hold_out_ids(self, id_list: list):
        """Set the input for the Pipeline

        Args:
           id_list (list): The list of hold out ids
        """
        self.pipeline["feature_set"]["hold_out_ids"] = id_list
```

Running a pipeline creates and deploys a set of Workbench Artifacts, DataSource, FeatureSet, Model and Endpoint. These artifacts can be viewed in the Sagemaker Console/Notebook interfaces or in the Workbench Dashboard UI.

!!! note "Not Finding a particular method?"
    The Workbench API Classes use the 'Core' Classes Internally, so for an extensive listing of all the methods available please take a deep dive into: [Workbench Core Classes](../core_classes/overview.md)
