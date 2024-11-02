"""Meta: A class that provides high level information and summaries of Cloud Platform Artifacts.
The Meta class provides 'account' information, configuration, etc. It also provides metadata for
AWS Artifacts, such as Data Sources, Feature Sets, Models, and Endpoints.
"""


class Meta:
    """Meta: A class that provides metadata functionality for Cloud Platform Artifacts.

    Common Usage:
       ```python
       from sageworks.api.meta import AbstractMeta
       meta = AbstractMeta()

       # Get the AWS Account Info
       meta.account()
       meta.cm()

       # These are 'list' methods
       meta.etl_jobs()
       meta.data_sources()
       meta.feature_sets()
       meta.models()
       meta.endpoints()
       meta.pipelines()
       meta.views()

       # These are 'describe' methods
       meta.data_source("abalone_data")
       meta.feature_set("abalone_features")
       meta.model("abalone-regression")
       meta.endpoint("abalone-endpoint")
       ```
    """

    def __new__(cls, platform="aws", *args, **kwargs):
        if platform == "aws":
            from sageworks.core.cloud_platform.aws.aws_meta import AWSMeta

            return AWSMeta(*args, **kwargs)
        elif platform == "azure":
            pass  # return AzureMeta(*args, **kwargs)
        elif platform == "gcp":
            pass  # return GCPMeta(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported platform: {platform}")


if __name__ == "__main__":
    """Exercise the SageWorks Meta Class"""
    import pandas as pd
    from pprint import pprint

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
    pprint(meta.models())

    # Get the Endpoints
    print("\n\n*** Endpoints ***")
    pprint(meta.endpoints())

    # Get the Pipelines
    print("\n\n*** Pipelines ***")
    pprint(meta.pipelines())

    # Now do a deep dive on all the Artifacts
    print("\n\n#")
    print("# Deep Dives ***")
    print("#")
