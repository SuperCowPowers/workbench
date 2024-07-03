import boto3

# Initialize Boto3 Glue client
glue_client = boto3.client("glue")

# Step 1: Get the existing workflow details
existing_workflow_name = "Workflow"
response = glue_client.get_workflow(Name=existing_workflow_name, IncludeGraph=True)
existing_workflow = response["Workflow"]

# Assume the workflow structure, jobs, and triggers need to be duplicated;
# this example focuses on the workflow duplication and trigger creation.

# Step 2: Create a new workflow (duplicate of the existing one)
new_workflow_name = "Workflow_Test"
glue_client.create_workflow(Name=new_workflow_name, Description="Test Workflow")

# Step 3: Create an 'on-demand' trigger for the new workflow
trigger_name = "on_demand_trigger_for_" + new_workflow_name
glue_client.create_trigger(
    Name=trigger_name,
    WorkflowName=new_workflow_name,
    Type="ON_DEMAND",  # This specifies the trigger is 'on-demand'
    Actions=[
        {
            # Define actions here; typically starting a job or crawler.
            # Example: 'JobName': 'your_glue_job_name'
            # This part depends on what the first step in your workflow is.
        }
    ],
)

print(f"New workflow '{new_workflow_name}' created with an 'on-demand' trigger.")
