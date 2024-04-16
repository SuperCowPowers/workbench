from pprint import pprint
from sageworks.api.pipeline_manager import PipelineManager

 # Create a PipelineManager
my_manager = PipelineManager()

# List the Pipelines
print("Listing Pipelines...")
pprint(my_manager.list_pipelines())

# Create a Pipeline from an Endpoint
abalone_pipeline = my_manager.create_from_endpoint("abalone-regression-end")

# Save the Pipeline
my_manager.save_pipeline("abalone_pipeline_v1", abalone_pipeline)
