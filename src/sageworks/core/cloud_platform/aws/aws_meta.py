"""AWSMeta: A class that provides high level information and summaries of AWS Platform Artifacts.
The AWSMeta class provides 'account' information, configuration, etc. It also provides metadata for
AWS Artifacts, such as Data Sources, Feature Sets, Models, and Endpoints.
"""

import logging
from typing import Union
import pandas as pd
import awswrangler as wr
from collections import defaultdict


# SageWorks Imports
from sageworks.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from sageworks.utils.config_manager import ConfigManager
from sageworks.utils.datetime_utils import datetime_string
from sageworks.utils.aws_utils import not_found_returns_none, aws_throttle, aws_tags_to_dict


class AWSMeta:
    """AWSMeta: A class that provides Metadata for a broad set of AWS Platform Artifacts

    Note: This is an internal class, the public API for this class is the 'Meta' class.
          Please see the 'Meta' class for more information.
    """

    def __init__(self):
        """AWSMeta Initialization"""
        self.log = logging.getLogger("sageworks")

        # Account and Configuration
        self.account_clamp = AWSAccountClamp()
        self.cm = ConfigManager()

        # Storing the size of various metadata for tracking
        self.metadata_sizes = defaultdict(dict)

        # Fill in AWS Specific Information
        self.sageworks_bucket = self.cm.get_config("SAGEWORKS_BUCKET")
        self.incoming_bucket = "s3://" + self.sageworks_bucket + "/incoming-data/"
        self.boto3_session = self.account_clamp.boto3_session
        self.sm_client = self.account_clamp.sagemaker_client()
        self.sm_session = self.account_clamp.sagemaker_session()

    def account(self) -> dict:
        """Cloud Platform Account Info

        Returns:
            dict: Cloud Platform Account Info
        """
        return self.account_clamp.get_aws_account_info()

    def config(self) -> dict:
        """Return the current SageWorks Configuration

        Returns:
            dict: The current SageWorks Configuration
        """
        return self.cm.get_all_config()

    def incoming_data(self) -> pd.DataFrame:
        """Get summary about the incoming raw data.

        Returns:
            pd.DataFrame: A summary of the incoming raw data in the S3 bucket.
        """
        self.log.debug(f"Refreshing metadata for S3 Bucket: {self.incoming_bucket}...")
        s3_file_info = self.s3_describe_objects(self.incoming_bucket)

        # Check if our bucket does not exist
        if s3_file_info is None:
            return pd.DataFrame()

        # Summarize the data into a DataFrame
        data_summary = []
        for full_path, info in s3_file_info.items():
            name = "/".join(full_path.split("/")[-2:]).replace("incoming-data/", "")
            size_mb = f"{info.get('ContentLength', 0) / 1_000_000:.2f} MB"

            summary = {
                "Name": name,
                "Size": size_mb,
                "Modified": datetime_string(info.get("LastModified", "-")),
                "ContentType": info.get("ContentType", "-"),
                "Encryption": info.get("ServerSideEncryption", "-"),
                "Tags": str(info.get("tags", "-")),  # Ensure 'tags' exist if needed
                "_aws_url": self.s3_to_console_url(full_path),
            }
            data_summary.append(summary)
        return pd.DataFrame(data_summary).convert_dtypes()

    def etl_jobs(self) -> pd.DataFrame:
        """Get summary data about Extract, Transform, Load (ETL) Jobs (AWS Glue Jobs)

        Returns:
            pd.DataFrame: A summary of the ETL Jobs deployed in the AWS Platform
        """

        # Retrieve Glue job metadata
        glue_client = self.boto3_session.client("glue")
        response = glue_client.get_jobs()
        jobs = response["Jobs"]

        # Extract relevant data for each job
        job_summary = []
        for job in jobs:
            job_name = job["Name"]
            job_runs = glue_client.get_job_runs(JobName=job_name, MaxResults=1)["JobRuns"]

            last_run = job_runs[0] if job_runs else None
            summary = {
                "Name": job_name,
                "Workers": job.get("NumberOfWorkers", "-"),
                "WorkerType": job.get("WorkerType", "-"),
                "Start Time": datetime_string(last_run["StartedOn"]) if last_run else "-",
                "Duration": f"{last_run['ExecutionTime']} sec" if last_run else "-",
                "State": last_run["JobRunState"] if last_run else "-",
                "_aws_url": self.glue_job_console_url(job_name),
            }
            job_summary.append(summary)

        return pd.DataFrame(job_summary).convert_dtypes()

    def data_sources(self) -> pd.DataFrame:
        """Get a summary of the Data Sources deployed in the AWS Platform

        Returns:
            pd.DataFrame: A summary of the Data Sources deployed in the AWS Platform
        """
        return self._list_catalog_tables("sageworks")

    def views(self, database: str = "sageworks") -> pd.DataFrame:
        """Get a summary of the all the Views, for the given database, in AWS

        Args:
            database (str, optional): Glue database. Defaults to 'sageworks'.

        Returns:
            pd.DataFrame: A summary of all the Views, for the given database, in AWS
        """
        summary = self._list_catalog_tables(database, views=True)
        summary.drop(columns=["Owner"], inplace=True, errors="ignore")  # Drop the Owner column
        return summary

    def feature_sets(self, details: bool = False) -> pd.DataFrame:
        """Get a summary of the Feature Sets in AWS.

        Args:
            details (bool, optional): Get additional details (Defaults to False).

        Returns:
            pd.DataFrame: A summary of the Feature Sets in AWS.
        """
        # Initialize the SageMaker paginator for listing feature groups
        paginator = self.sm_client.get_paginator("list_feature_groups")
        data_summary = []

        # Use the paginator to retrieve all feature groups
        for page in paginator.paginate():
            for fg in page["FeatureGroupSummaries"]:
                name = fg["FeatureGroupName"]

                # Get details if requested
                feature_set_details = {}
                if details:
                    feature_set_details.update(self.sm_client.describe_feature_group(FeatureGroupName=name))

                # Retrieve SageWorks metadata from tags
                aws_tags = self.get_aws_tags(fg["FeatureGroupArn"])
                summary = {
                    "Feature Group": name,
                    "Owner": aws_tags.get("sageworks_owner", "-"),
                    "Created": datetime_string(feature_set_details.get("CreationTime")),
                    "Num Columns": len(feature_set_details.get("FeatureDefinitions", [])),
                    "Input": aws_tags.get("sageworks_input", "-"),
                    "Tags": aws_tags.get("sageworks_tags", "-"),
                    "Online": str(feature_set_details.get("OnlineStoreConfig", {}).get("EnableOnlineStore", "Unknown")),
                    "Offline": "True" if feature_set_details.get("OfflineStoreConfig") else "Unknown",
                    "_aws_url": self.feature_group_console_url(name),
                }
                data_summary.append(summary)

        # Return the summary as a DataFrame
        return pd.DataFrame(data_summary).convert_dtypes()

    def models(self, details: bool = False) -> pd.DataFrame:
        """Get a summary of the Models in AWS.

        Args:
            details (bool, optional): Get additional details (Defaults to False).

        Returns:
            pd.DataFrame: A summary of the Models in AWS.
        """
        # Initialize the SageMaker paginator for listing model package groups
        paginator = self.sm_client.get_paginator("list_model_package_groups")
        model_summary = []

        # Use the paginator to retrieve all model package groups
        for page in paginator.paginate():
            for group in page["ModelPackageGroupSummaryList"]:
                model_group_name = group["ModelPackageGroupName"]
                created = datetime_string(group["CreationTime"])
                description = group.get("ModelPackageGroupDescription", "-")

                # Initialize variables for details retrieval
                model_details = {}
                aws_tags = {}
                status = "Unknown"
                health_tags = ""

                # If details=True get the latest model package details
                if details:
                    latest_model = self.get_latest_model_package_info(model_group_name)
                    if latest_model:
                        model_details.update(
                            self.sm_client.describe_model_package(ModelPackageName=latest_model["ModelPackageArn"])
                        )
                        aws_tags = self.get_aws_tags(group["ModelPackageGroupArn"])
                        health_tags = aws_tags.get("sageworks_health_tags", "")
                        status = model_details.get("ModelPackageStatus", "Unknown")
                    else:
                        health_tags = "model_not_found"
                        status = "No Models"

                # Compile model summary
                summary = {
                    "Model Group": model_group_name,
                    "Health": health_tags,
                    "Owner": aws_tags.get("sageworks_owner", "-"),
                    "Model Type": aws_tags.get("sageworks_model_type", "-"),
                    "Created": created,
                    "Ver": model_details.get("ModelPackageVersion", "-"),
                    "Tags": aws_tags.get("sageworks_tags", "-"),
                    "Input": aws_tags.get("sageworks_input", "-"),
                    "Status": status,
                    "Description": description,
                    "_aws_url": self.model_package_group_console_url(model_group_name),
                }
                model_summary.append(summary)

        # Return the summary as a DataFrame
        return pd.DataFrame(model_summary).convert_dtypes()

    def endpoints(self, refresh: bool = False) -> pd.DataFrame:
        """Get a summary of the Endpoints in AWS.

        Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            pd.DataFrame: A summary of the Endpoints in AWS.
        """
        # Initialize the SageMaker client and list all endpoints
        sagemaker_client = self.boto3_session.client("sagemaker")
        paginator = sagemaker_client.get_paginator("list_endpoints")
        data_summary = []

        # Use the paginator to retrieve all endpoints
        for page in paginator.paginate():
            for endpoint in page["Endpoints"]:
                endpoint_name = endpoint["EndpointName"]
                endpoint_info = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)

                # Retrieve SageWorks metadata from tags
                sageworks_meta = self.get_aws_tags(endpoint_info["EndpointArn"])
                health_tags = sageworks_meta.get("sageworks_health_tags", "")

                # Retrieve endpoint configuration to determine instance type or serverless info
                endpoint_config_name = endpoint_info["EndpointConfigName"]
                endpoint_config = sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
                production_variant = endpoint_config["ProductionVariants"][0]

                # Determine instance type or serverless configuration
                instance_type = production_variant.get("InstanceType")
                if instance_type is None:
                    # If no instance type, it's a serverless configuration
                    mem_size = production_variant["ServerlessConfig"]["MemorySizeInMB"]
                    concurrency = production_variant["ServerlessConfig"]["MaxConcurrency"]
                    instance_type = f"Serverless ({mem_size//1024}GB/{concurrency})"

                # Compile endpoint summary
                summary = {
                    "Name": endpoint_name,
                    "Health": health_tags,
                    "Instance": instance_type,
                    "Created": datetime_string(endpoint_info.get("CreationTime")),
                    "Tags": sageworks_meta.get("sageworks_tags", "-"),
                    "Input": sageworks_meta.get("sageworks_input", "-"),
                    "Status": endpoint_info["EndpointStatus"],
                    "Variant": production_variant.get("VariantName", "-"),
                    "Capture": str(endpoint_info.get("DataCaptureConfig", {}).get("EnableCapture", "False")),
                    "Samp(%)": str(endpoint_info.get("DataCaptureConfig", {}).get("CurrentSamplingPercentage", "-")),
                }
                data_summary.append(summary)

        # Return the summary as a DataFrame
        return pd.DataFrame(data_summary).convert_dtypes()

    @not_found_returns_none
    def glue_job(self, job_name: str) -> Union[dict, None]:
        """Describe a single Glue ETL Job in AWS.

        Args:
            job_name (str): The name of the Glue job to describe.

        Returns:
            dict: A detailed description of the Glue job (None if not found).
        """
        glue_client = self.boto3_session.client("glue")
        job_details = glue_client.get_job(JobName=job_name)["Job"]
        return {
            "Job Name": job_details["Name"],
            "Worker Type": job_details.get("WorkerType", "-"),
            "Number of Workers": job_details.get("NumberOfWorkers", "-"),
            "Last Modified": datetime_string(job_details.get("LastModifiedOn")),
            "Role": job_details.get("Role", "-"),
            "Description": job_details.get("Description", "-"),
            "Tags": job_details.get("Tags", {}),
        }

    @not_found_returns_none
    def data_source(self, table_name: str, database: str = "sageworks") -> Union[dict, None]:
        """Describe a single Data Source (Glue Table) in AWS.

        Args:
            table_name (str): The name of the Glue table (data source) to describe.
            database (str, optional): The Glue database where the table is located. Defaults to 'sageworks'.

        Returns:
            dict: A detailed description of the data source (None if not found).
        """
        # Retrieve table metadata from the Glue catalog
        glue_client = self.boto3_session.client("glue")
        table_details = glue_client.get_table(DatabaseName=database, Name=table_name)["Table"]
        return table_details

    @not_found_returns_none
    def feature_set(self, feature_group_name: str) -> Union[dict, None]:
        """Describe a single Feature Set (Feature Group) in AWS.

        Args:
            feature_group_name (str): The name of the feature group to describe.

        Returns:
            dict: A detailed description of the feature group (None if not found).
        """
        feature_set_details = self.sm_client.describe_feature_group(FeatureGroupName=feature_group_name)

        # Retrieve SageWorks metadata from the AWS tags
        sageworks_meta = self.get_aws_tags(arn=feature_set_details["FeatureGroupArn"])
        add_data = {
            "athena_database": self._athena_database_name(feature_set_details),
            "athena_table": self._athena_table_name(feature_set_details),
            "s3_storage": self._s3_storage(feature_set_details),
        }
        sageworks_meta.update(add_data)
        feature_set_details["sageworks_meta"] = sageworks_meta
        return feature_set_details

    @not_found_returns_none
    def stand_alone_model(self, model_name: str) -> Union[dict, None]:
        """Describe a single Model in AWS.

        Args:
            model_name (str): The name of the model to describe.

        Returns:
            dict: A detailed description of the model (None if not found).
        """
        model_details = self.sm_client.describe_model(ModelName=model_name)
        return model_details

    @not_found_returns_none
    def model(self, model_group_name: str) -> Union[dict, None]:
        """Describe a single Model Package Group in AWS.

        Args:
            model_group_name (str): The name of the model package group to describe.

        Returns:
            dict: A detailed description of the model package group (None if not found).
        """
        # Retrieve Model Package Group details
        model_group_details = self.sm_client.describe_model_package_group(ModelPackageGroupName=model_group_name)
        model_package_group_arn = model_group_details["ModelPackageGroupArn"]

        # Retrieve the list of model package ARNs
        model_package_arns = [
            package["ModelPackageArn"]
            for package in self.sm_client.list_model_packages(ModelPackageGroupName=model_group_name)[
                "ModelPackageSummaryList"
            ]
        ]

        # Get detailed information for each model package and add to the list
        model_group_details["ModelPackageList"] = [
            self.sm_client.describe_model_package(ModelPackageName=arn) for arn in model_package_arns
        ]

        # Retrieve SageWorks metadata from AWS tags
        model_group_details["sageworks_meta"] = self.get_aws_tags(model_package_group_arn)
        return model_group_details

    @not_found_returns_none
    def endpoint(self, endpoint_name: str) -> Union[dict, None]:
        """Describe a single Endpoint in AWS.

        Args:
            endpoint_name (str): The name of the endpoint to describe.

        Returns:
            dict: A detailed description of the endpoint (None if not found).
        """
        endpoint_details = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config_name = endpoint_details.get("EndpointConfigName")
        endpoint_config = self.sm_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        production_variant = endpoint_config["ProductionVariants"][0]

        instance_type = production_variant.get("InstanceType")
        if instance_type is None:
            # If no instance type, it's a serverless configuration
            mem_size = production_variant["ServerlessConfig"]["MemorySizeInMB"]
            concurrency = production_variant["ServerlessConfig"]["MaxConcurrency"]
            instance_type = f"Serverless ({mem_size//1024}GB/{concurrency})"
        endpoint_details["InstanceType"] = instance_type

        # Retrieve SageWorks metadata from AWS tags
        endpoint_details["sageworks_meta"] = self.get_aws_tags(endpoint_details["EndpointArn"])
        return endpoint_details

    # These are helper methods to construct the AWS URL for the Artifacts
    @staticmethod
    def s3_to_console_url(s3_path: str) -> str:
        """Convert an S3 path to a clickable AWS Console URL."""
        bucket, key = s3_path.replace("s3://", "").split("/", 1)
        return f"https://s3.console.aws.amazon.com/s3/object/{bucket}?prefix={key}"

    def glue_job_console_url(self, job_name: str) -> str:
        """Convert a Glue job name and region into a clickable AWS Console URL."""
        region = self.account_clamp.region
        return f"https://{region}.console.aws.amazon.com/gluestudio/home?region={region}#/editor/job/{job_name}/runs"

    def data_catalog_console_url(self, table_name: str, database: str) -> str:
        """Convert a database and table name to a clickable Athena Console URL."""
        region = self.boto3_session.region_name
        aws = "console.aws.amazon.com"
        return f"https://{region}.{aws}/athena/home?region={region}#query/databases/{database}/tables/{table_name}"

    def feature_group_console_url(self, group_name: str) -> str:
        """Generate an AWS Console URL for a given Feature Group."""
        region = self.boto3_session.region_name
        aws = "console.aws.amazon.com"
        return f"https://{region}.{aws}/sagemaker/home?region={region}#/feature-groups/{group_name}/details"

    def model_package_group_console_url(self, group_name: str) -> str:
        """Generate an AWS Console URL for a given Model Package Group."""
        region = self.boto3_session.region_name
        aws = "console.aws.amazon.com"
        return f"https://{region}.{aws}.com/sagemaker/home?region={region}#/model-registry/{group_name}/details"

    def endpoint_console_url(self, endpoint_name: str) -> str:
        """Generate an AWS Console URL for a given Endpoint."""
        region = self.boto3_session.region_name
        aws = "console.aws.amazon.com"
        return f"https://{region}.{aws}/sagemaker/home?region={region}#/endpoints/{endpoint_name}/details"

    # Helper methods to pull specific data out of the AWS Feature Group metadata
    def _athena_database_name(self, feature_group_info: dict) -> Union[str, None]:
        """Internal: Get the Athena Database Name for a specific feature group"""
        try:
            return feature_group_info["OfflineStoreConfig"]["DataCatalogConfig"]["Database"].lower()
        except KeyError:
            feature_group_name = feature_group_info.get("FeatureGroupName", "Unknown")
            self.log.critical(f"Could not find OfflineStore Database for {feature_group_name}!")
            return None

    def _athena_table_name(self, feature_group_info: dict) -> Union[str, None]:
        """Internal: Get the Athena Database Name for a specific feature group"""
        try:
            return feature_group_info["OfflineStoreConfig"]["DataCatalogConfig"]["TableName"].lower()
        except KeyError:
            feature_group_name = feature_group_info.get("FeatureGroupName", "Unknown")
            self.log.critical(f"Could not find OfflineStore Table for {feature_group_name}!")
            return None

    @staticmethod
    def _s3_storage(feature_group_info: dict) -> str:
        """Internal: Get the S3 Location for a specific feature group"""
        return feature_group_info["OfflineStoreConfig"]["S3StorageConfig"]["ResolvedOutputS3Uri"]

    @not_found_returns_none
    def s3_describe_objects(self, bucket: str) -> Union[dict, None]:
        """Internal: Get the S3 File Information for the given bucket"""
        return wr.s3.describe_objects(path=bucket, boto3_session=self.boto3_session)

    @aws_throttle
    def get_aws_tags(self, arn: str) -> Union[dict, None]:
        """List the tags for the given AWS ARN"""

        # Sanity check the ARN
        if arn is None:
            self.log.error("ARN is None, cannot retrieve tags.")
            return None

        # Grab the tags from AWS
        return aws_tags_to_dict(self.sm_session.list_tags(resource_arn=arn))

    @aws_throttle
    def get_latest_model_package_info(self, model_group_name: str) -> Union[dict, None]:
        """Get the latest model package information for the given model group.

        Args:
            model_group_name (str): The name of the model package group.

        Returns:
            dict: The latest model package information.
        """
        model_package_list = self.sm_client.list_model_packages(
            ModelPackageGroupName=model_group_name,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,  # Get only the latest model
        )
        # If no model packages are found, return None
        if not model_package_list["ModelPackageSummaryList"]:
            return None

        # Return the latest model package
        return model_package_list["ModelPackageSummaryList"][0]

    def _list_catalog_tables(self, database: str, views: bool = False) -> pd.DataFrame:
        """Internal method to retrieve and summarize Glue catalog tables or views.

        Args:
            database (str): The Glue catalog database name.
            views (bool): If True, filter for views (VIRTUAL_VIEW), otherwise for tables.

        Returns:
            pd.DataFrame: A summary of the tables or views in the specified database.
        """
        self.log.debug(f"Data Catalog Database: {database} for {'views' if views else 'tables'}...")
        all_tables = list(wr.catalog.get_tables(database=database, boto3_session=self.boto3_session))

        # Filter based on whether we're looking for views or tables
        table_type = "VIRTUAL_VIEW" if views else "EXTERNAL_TABLE"
        filtered_tables = [
            table for table in all_tables if not table["Name"].startswith("_") and table["TableType"] == table_type
        ]

        # Summarize the data in a DataFrame
        data_summary = []
        for table in filtered_tables:
            summary = {
                "Name": table["Name"],
                "Owner": table.get("Parameters", {}).get("sageworks_owner", "-"),
                "Database": database,
                "Modified": datetime_string(table["UpdateTime"]),
                "Tags": table.get("Parameters", {}).get("sageworks_tags", "-"),
                "Columns": len(table["StorageDescriptor"].get("Columns", [])),
                "Input": str(
                    table.get("Parameters", {}).get("sageworks_input", "-"),
                ),
                "_aws_url": self.data_catalog_console_url(table["Name"], database),
            }
            data_summary.append(summary)

        return pd.DataFrame(data_summary).convert_dtypes()

    def close(self):
        """Close the AWSMeta Class"""
        self.log.debug("Closing the AWSMeta Class")

    def __repr__(self):
        return f"AWSMeta({self.account_clamp.account_id}: {self.account_clamp.region})"


if __name__ == "__main__":
    """Exercise the SageWorks AWSMeta Class"""
    from pprint import pprint
    import time

    # Pandas Display Options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Create the class
    meta = AWSMeta()

    # Test the __repr__ method
    print(meta)

    # Get the AWS Account Info
    print("*** AWS Account ***")
    pprint(meta.account())

    # Get the SageWorks Configuration
    print("*** SageWorks Configuration ***")
    pprint(meta.config())

    # Get the Incoming Data
    print("\n\n*** Incoming Data ***")
    print(meta.incoming_data())

    # Get the AWS Glue Jobs (ETL Jobs)
    print("\n\n*** ETL Jobs ***")
    print(meta.etl_jobs())

    # Get the Data Sources
    print("\n\n*** Data Sources ***")
    print(meta.data_sources())

    # Get the Views (Data Sources)
    print("\n\n*** Views (Data Sources) ***")
    print(meta.views("sageworks"))

    # Get the Views (Feature Sets)
    print("\n\n*** Views (Feature Sets) ***")
    fs_views = meta.views("sagemaker_featurestore")
    print(fs_views)

    # Get the Feature Sets
    print("\n\n*** Feature Sets ***")
    pprint(meta.feature_sets())

    # Get the Models
    print("\n\n*** Models ***")
    start_time = time.time()
    pprint(meta.models())
    print(f"Elapsed Time Model (no details): {time.time() - start_time:.2f}")

    # Get the Models with Details
    print("\n\n*** Models with Details ***")
    start_time = time.time()
    pprint(meta.models(details=True))
    print(f"Elapsed Time Model (with details): {time.time() - start_time:.2f}")

    # Get the Endpoints
    print("\n\n*** Endpoints ***")
    pprint(meta.endpoints())

    # Test out the specific artifact details methods
    print("\n\n*** Glue Job Details ***")
    pprint(meta.glue_job("Glue_Job_1"))
    print("\n\n*** DataSource Details ***")
    pprint(meta.data_source("abalone_data"))
    print("\n\n*** FeatureSet Details ***")
    pprint(meta.feature_set("abalone_features"))
    # print("\n\n*** StandAlone Model Details ***")
    # pprint(meta.stand_alone_model("tbd"))
    print("\n\n*** Model Details ***")
    pprint(meta.model("abalone-regression"))
    print("\n\n*** Endpoint Details ***")
    pprint(meta.endpoint("abalone-regression-end"))
    pprint(meta.endpoint("test-timing-realtime"))

    # Test out a non-existent model
    print("\n\n*** Model Doesn't Exist ***")
    pprint(meta.model("non-existent-model"))
