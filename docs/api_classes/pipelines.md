# Pipelines

!!! tip inline end "Pipeline Examples"
    Examples of using the Pipeline classes are listed at the bottom of this page [Examples](#examples).
    
Pipelines are a sequence of SageWorks transforms. So if you have a nightly ML workflow you can capture that as a Pipeline. Here's an example pipeline:

```py title=nightly_sol_pipeline_v1.json
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

::: sageworks.api.pipeline_manager

::: sageworks.api.pipeline


## Examples

**Make a Pipeline**

Pipelines are just JSON files (see `sageworks/examples/pipelines/`). You can copy one and make changes to fit your objects/use case, or if you have a set of SageWorks artifacts created you can 'backtrack' from the Endpoint and have it create the Pipeline for you.

```py title="pipeline_manager.py"
from sageworks.api.pipeline_manager import PipelineManager

 # Create a PipelineManager
my_manager = PipelineManager()

# List the Pipelines
pprint(my_manager.list_pipelines())

# Create a Pipeline from an Endpoint
abalone_pipeline = my_manager.create_from_endpoint("abalone-regression-end")

# Save the Pipeline
my_manager.save_pipeline("abalone_pipeline_v1", abalone_pipeline)
```

**Output**

```py
Listing Pipelines...
[{'last_modified': datetime.datetime(2024, 4, 16, 21, 10, 6, tzinfo=tzutc()),
  'name': 'abalone_pipeline_v1',
  'size': 445}]
```
**Pipeline Details**

!!!tip inline end "The details() method"
    The `detail()` method on the Pipeline class provides a lot of useful information. All of the SageWorks classes have a `details()` method try it out!

```py title="pipeline_details.py"
from sageworks.api.pipeline import Pipeline

# Retrieve an existing Pipeline
my_pipeline = Pipeline("abalone_pipeline_v1")
pprint(my_pipeline.details(recompute=True))
```

**Output**

```py
{'name': 'abalone_pipeline_v1',
 'pipeline': {'data_source': {'input': '/Users/briford/work/sageworks/data/abalone.csv',
                              'name': 'abalone_data',
                              'tags': ['abalone_data']},
              'endpoint': {'input': 'abalone-regression',
                           'name': 'abalone-regression-end',
                           'tags': ['abalone', 'regression']},
              'feature_set': {'input': 'abalone_data',
                              'name': 'abalone_features',
                              'tags': ['abalone_features']},
              'model': {'input': 'abalone_features',
                        'name': 'abalone-regression',
                        'tags': ['abalone', 'regression']}},
 's3_path': 's3://sandbox-sageworks-artifacts/pipelines/abalone_pipeline_v1.json'}
```

**Pipeline Execution**

!!!tip inline end "Executing the Pipeline is obviously the most important reason for creating one. If gives you a reproducible way to capture, inspect, and run the same ML pipeline on different data (nightly).

```py title="pipeline_execution.py"
from sageworks.api.pipeline import Pipeline

# Retrieve an existing Pipeline
my_pipeline = Pipeline("abalone_pipeline_v1")
pprint(my_pipeline.details(recompute=True))

# Execute the Pipeline
my_pipeline.execute()
```

**Output**

```py
TBD
```

## SageWorks UI
Running a pipeline creates and deploys a set of SageWorks Artifacts, DataSource, FeatureSet, Model and Endpoint. These artifacts can be viewed in the Sagemaker Console/Notebook interfaces or in the SageWorks Dashboard UI.

<figure>
<img alt="sageworks_endpoints" src="https://github.com/SuperCowPowers/sageworks/assets/4806709/b5eab741-2c23-4c5e-9495-15fd3ea8155c">
<figcaption>SageWorks Dashboard: Endpoints</figcaption>
</figure>


!!! note "Not Finding a particular method?"
    The SageWorks API Classes use the 'Core' Classes Internally, so for an extensive listing of all the methods available please take a deep dive into: [SageWorks Core Classes](../core_classes/overview.md)
