"""Class: GlueJobs"""

# SageWorks Imports
from sageworks.aws_service_broker.aws_service_connectors.connector import Connector
from sageworks.utils.aws_utils import compute_size

# References
# - https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming-python-calling.html


class GlueJobs(Connector):
    """GlueJobs: Helper Class for tracking AWS Glue Jobs"""

    def __init__(self):
        """GlueJobs: Helper Class for tracking AWS Glue Jobs"""
        # Call SuperClass Initialization
        super().__init__()

        # Set up our glue client and internal data storage
        self.glue_client = self.boto_session.client("glue")
        self.glue_job_metadata = {}

    def check(self) -> bool:
        """Check if we can reach/connect to this AWS Service"""
        try:
            self.glue_client.get_jobs()
            return True
        except Exception as e:
            self.log.critical(f"Error connecting to AWS Glue: {e}")
            return False

    def refresh(self):
        """Refresh the metadata for all the Glue Jobs"""
        self.log.info("Refreshing Glue Job Metadata from AWS Glue...")

        # For each Glue Job get the detailed metadata about that job
        jobs = self.glue_client.get_jobs()
        job_names = [job["Name"] for job in jobs["Jobs"]]

        # Iterate through each job and get the details
        for job_name in job_names:
            # Join the General Job and Last Run Info
            job_info = self.glue_client.get_job(JobName=job_name)["Job"]
            last_run_info = self._get_last_run_info(job_name)

            # Add the job info to our internal data storage
            self.glue_job_metadata[job_name] = job_info
            self.glue_job_metadata[job_name]["sageworks_meta"] = last_run_info

        # Track the size of the metadata
        for key in self.glue_job_metadata.keys():
            self.metadata_size_info[key] = compute_size(self.glue_job_metadata[key])

    def summary(self) -> dict:
        """Return a summary of all the AWS Glue Jobs"""
        return self.glue_job_metadata

    def get_glue_jobs(self) -> list:
        """Get the glue job names for the AWS Glue Jobs"""
        return list(self.glue_job_metadata.keys())

    def _get_last_run_info(self, job_name: str) -> dict:
        """Internal: Get the last run status for the given glue job"""
        # Get job runs
        job_runs = self.glue_client.get_job_runs(JobName=job_name)

        # Check if there are any runs
        if job_runs["JobRuns"]:
            # Get the most recent run and return the state and start time
            last_run = job_runs["JobRuns"][0]
            return {
                "status": last_run["JobRunState"],
                "last_run": last_run["StartedOn"],
            }
        else:
            return {"status": "-", "last_run": "-"}

    def _get_job_arn(self, job_name: str) -> dict:
        """Internal: Get the AWS ARN for the given Glue Job name"""

        # Construct the ARN for this Glue Job
        region = self.aws_account_clamp.region
        account_id = self.aws_account_clamp.account_id
        job_arn = f"arn:aws:glue:{region}:{account_id}:job/{job_name}"
        return job_arn


if __name__ == "__main__":
    from pprint import pprint

    # Create the class and get the AWS Glue Job Info
    glue_info = GlueJobs()

    # The connectors need an explicit refresh to populate themselves
    glue_info.refresh()

    # Get a summary of the AWS Glue Jobs
    pprint(glue_info.summary())

    # List Glue Jobs
    for my_job_name in glue_info.get_glue_jobs():
        print(f"{my_job_name}")

    # Print out the metadata sizes for this connector
    pprint(glue_info.get_metadata_sizes())
