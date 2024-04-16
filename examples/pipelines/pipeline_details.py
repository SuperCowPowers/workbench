from pprint import pprint
from sageworks.api.pipeline import Pipeline

# Retrieve an existing Pipeline
my_pipeline = Pipeline("abalone_pipeline_v1")
pprint(my_pipeline.details(recompute=True))
