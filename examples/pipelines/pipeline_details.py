import json
from sageworks.api.pipeline import Pipeline

# Retrieve an existing Pipeline
my_pipeline = Pipeline("abalone_pipeline_v1")
print(json.dumps(my_pipeline.details(), indent=4))
