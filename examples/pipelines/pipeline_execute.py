from sageworks.api.pipeline import Pipeline

# Retrieve an existing Pipeline
my_pipeline = Pipeline("abalone_pipeline_v1")

# Execute the Pipeline
# my_pipeline.execute()  # Full execution

# Partial executions
my_pipeline.execute_partial(["data_source", "feature_set"])
# my_pipeline.execute_partial(["model", "endpoint"])
