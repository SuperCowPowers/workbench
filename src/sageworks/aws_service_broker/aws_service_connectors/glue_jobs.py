"""Class: GlueJobs"""
import boto3

# SageWorks Imports
from sageworks.aws_service_broker.aws_service_connectors.connector import Connector

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

    def refresh_impl(self):
        """Load/reload all the metadata for all the Glue Jobs"""

        # For each Glue Job get the detailed metadata about that job
        jobs = self.glue_client.get_jobs()
        job_names = [job["Name"] for job in jobs["Jobs"]]

        # Iterate through each job and print details
        for job_name in job_names:
            job_info = self.glue_client.get_job(JobName=job_name)
            self.glue_job_metadata[job_name] = job_info["Job"]

    def aws_meta(self) -> dict:
        """Get ALL the AWS metadata for the AWS Glue Jobs"""
        return self.glue_job_metadata

    def get_glue_jobs(self) -> list:
        """Get the glue job names for the AWS Glue Jobs"""
        return list(self.glue_job_metadata.keys())

    def get_job_info(self, job_name: str) -> dict | None:
        """Get the information for the given glue job"""
        return self.glue_job_metadata.get(job_name)

    def get_job_arn(self, job_name: str) -> dict:
        """Get the AWS ARN for the given Glue Job name"""

        # Construct the ARN for this Glue Job
        region = self.aws_account_clamp.region()
        account_id = self.aws_account_clamp.account_id()
        job_arn = f"arn:aws:glue:{region}:{account_id}:job/{job_name}"
        return job_arn


if __name__ == "__main__":
    from pprint import pprint

    # Create the class and get the AWS Data Catalog database info
    glue_jobs = GlueJobs()

    # The connectors need an explicit refresh to populate themselves
    glue_jobs.refresh()

    # List Glue Jobs and their details
    for job_name in glue_jobs.get_glue_jobs():
        print(f"{job_name}")
        details = glue_jobs.get_job_info(job_name)
        pprint(details)

    # Get the ARN for a specific Glue Job
    arn = glue_jobs.get_job_arn(job_name)
    print(f"ARN for {job_name} is {arn}")
