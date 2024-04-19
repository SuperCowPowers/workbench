from pprint import pprint
from sageworks.api.pipeline_manager import PipelineManager

# Create a PipelineManager
my_manager = PipelineManager()

# List the Pipelines
print("Listing Pipelines...")
pprint(my_manager.list_pipelines())

# Create a Pipeline from an Endpoint
abalone_pipeline = my_manager.create_from_endpoint("abalone-regression-end")

# Publish the Pipeline
my_manager.publish_pipeline("abalone_pipeline_v1", abalone_pipeline)

# Save the Pipeline to a local file
my_manager.save_pipeline_to_file(abalone_pipeline, "abalone_pipeline_v1.json")
