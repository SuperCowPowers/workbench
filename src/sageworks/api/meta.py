"""Meta: An abstract class that provides high level information and summaries of Cloud Platform Artifacts.
The Meta class provides 'account' information, configuration, etc. It also provides metadata for Artifacts,
such as Data Sources, Feature Sets, Models, and Endpoints.
"""

import logging
from typing import Union
import pandas as pd


# SageWorks Imports
from sageworks.core.cloud_platform.aws.aws_meta import AWSMeta


class Meta(AWSMeta):
    """Meta: A class that provides metadata functionality for Cloud Platform Artifacts.

    Common Usage:
       ```python
       from sageworks.api.meta import Meta
       meta = Meta()

       # Get the AWS Account Info
       meta.account()
       meta.config()

       # These are 'list' methods
       meta.etl_jobs()
       meta.data_sources()
       meta.feature_sets(details=True/False)
       meta.models(details=True/False)
       meta.endpoints()
       meta.views()

       # These are 'describe' methods
       meta.data_source("abalone_data")
       meta.feature_set("abalone_features")
       meta.model("abalone-regression")
       meta.endpoint("abalone-endpoint")
       ```
    """

    def __init__(self, use_cache: bool = False):
        """Meta Initialization

        Args:
            use_cache (bool, optional): Use a cache for the metadata. Defaults to False.
        """
        self.log = logging.getLogger("sageworks")

        # Call the SuperClass Initialization
        super().__init__()

        # We will be caching the metadata?
        self.use_cache = use_cache

    def account(self) -> dict:
        """Cloud Platform Account Info

        Returns:
            dict: Cloud Platform Account Info
        """
        return super().account()

    def config(self) -> dict:
        """Return the current SageWorks Configuration

        Returns:
            dict: The current SageWorks Configuration
        """
        return super().config()
    
    def incoming_data(self) -> pd.DataFrame:
        """Get summary data about data in the incoming raw data

        Returns:
            pd.DataFrame: A summary of the incoming raw data
        """
        return super().incoming_data()

    def etl_jobs(self) -> pd.DataFrame:
        """Get summary data about Extract, Transform, Load (ETL) Jobs

        Returns:
            pd.DataFrame: A summary of the ETL Jobs deployed in the Cloud Platform
        """
        return super().etl_jobs()

    def data_sources(self) -> pd.DataFrame:
        """Get a summary of the Data Sources deployed in the Cloud Platform

        Returns:
            pd.DataFrame: A summary of the Data Sources deployed in the Cloud Platform
        """
        return super().data_sources()
    
    def views(self, database: str = "sageworks") -> pd.DataFrame:
        """Get a summary of the all the Views, for the given database, in AWS

        Args:
            database (str, optional): Glue database. Defaults to 'sageworks'.

        Returns:
            pd.DataFrame: A summary of all the Views, for the given database, in AWS
        """
        return super().views(database=database)
    
    def feature_sets(self, details: bool = False) -> pd.DataFrame:
        """Get a summary of the Feature Sets deployed in the Cloud Platform

        Args:
            details (bool, optional): Include detailed information. Defaults to False.

        Returns:
            pd.DataFrame: A summary of the Feature Sets deployed in the Cloud Platform
        """
        return super().feature_sets(details=details)
    
    def models(self, details: bool = False) -> pd.DataFrame:
        """Get a summary of the Models deployed in the Cloud Platform

        Args:
            details (bool, optional): Include detailed information. Defaults to False.

        Returns:
            pd.DataFrame: A summary of the Models deployed in the Cloud Platform
        """
        return super().models(details=details)

    def endpoints(self) -> pd.DataFrame:
        """Get a summary of the Endpoints deployed in the Cloud Platform

        Returns:
            pd.DataFrame: A summary of the Endpoints in the Cloud Platform
        """
        return super().endpoints()
    
    def glue_job(self, job_name: str) -> Union[dict, None]:
        """Get the details of a specific Glue Job

        Args:
            job_name (str): The name of the Glue Job

        Returns:
            dict: The details of the Glue Job (None if not found)
        """
        return super().glue_job(job_name=job_name)

    def data_source(self, data_source_name: str, database: str = "sageworks") -> Union[dict, None]:
        """Get the details of a specific Data Source

        Args:
            data_source_name (str): The name of the Data Source
            database (str, optional): The Glue database. Defaults to 'sageworks'.

        Returns:
            dict: The details of the Data Source (None if not found)
        """
        return super().data_source(table_name=data_source_name, database=database)

    def feature_set(self, feature_set_name: str) -> Union[dict, None]:
        """Get the details of a specific Feature Set

        Args:
            feature_set_name (str): The name of the Feature Set

        Returns:
            dict: The details of the Feature Set (None if not found)
        """
        return super().feature_set(feature_group_name=feature_set_name)
    
    def model(self, model_name: str) -> Union[dict, None]:
        """Get the details of a specific Model

        Args:
            model_name (str): The name of the Model

        Returns:
            dict: The details of the Model (None if not found)
        """
        return super().model(model_group_name=model_name)

    def endpoint(self, endpoint_name: str) -> Union[dict, None]:
        """Get the details of a specific Endpoint

        Args:
            endpoint_name (str): The name of the Endpoint

        Returns:
            dict: The details of the Endpoint (None if not found)
        """
        return super().endpoint(endpoint_name=endpoint_name)


if __name__ == "__main__":
    """Exercise the SageWorks AWSMeta Class"""
    from pprint import pprint
    import time

    # Pandas Display Options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Create the class
    meta = Meta()

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
