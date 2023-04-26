"""S3HeavyToDataSource: Class to move HEAVY S3 Files into a SageWorks DataSource"""
import sys
import boto3
from botocore.exceptions import ClientError

# Import all the AWS Glue Python Libraries
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext


class S3HeavyToDataSource:
    def __init__(self, glue_context: GlueContext, input_uuid: str, output_uuid: str):
        """S3HeavyToDataSource: Class to move HEAVY S3 Files into a SageWorks DataSource
        Args:
            glue_context: GlueContext, AWS Glue Specific wrapper around SparkContext
            input_uuid (str): The S3 Path to the files to be loaded
            output_uuid (str): The UUID of the SageWorks DataSource to be created
        """
        self.log = glue_context.get_logger()

        # FIXME: Pull these from Parameter Store or Config
        self.input_uuid = input_uuid
        self.output_uuid = output_uuid
        self.output_meta = {"sageworks_input": self.input_uuid}
        sageworks_bucket = "s3://scp-sageworks-artifacts"
        self.data_source_s3_path = sageworks_bucket + "/data-sources"

        # Our Spark Context
        self.glue_context = glue_context

    def transform(self, input_type: str = "json", overwrite: bool = True):
        """Convert the CSV or JSON data into Parquet Format in the SageWorks S3 Bucket, and
        store the information about the data to the AWS Data Catalog sageworks database"""

        # Add some tags here
        tags = ["heavy"]

        # Create the Output Parquet file S3 Storage Path
        s3_storage_path = f"{self.data_source_s3_path}/{self.output_uuid}"

        # Read JSONL files from S3 and infer schema dynamically
        self.log.info(f"Reading JSONL files from {self.input_uuid}...")
        input_dyf = self.glue_context.create_dynamic_frame.from_options(
            connection_type="s3",
            connection_options={
                "paths": [self.input_uuid],
                "recurse": True,
                "gzip": True,
            },
            format=input_type,
            # format_options={'jsonPath': 'auto'}, Look into this later
            transformation_ctx="apply_mapping",
        )
        self.log.info("Incoming DataFrame...")
        input_dyf.show(5)

        # Write Parquet files to S3
        self.log.info(f"Writing Parquet files to {s3_storage_path}...")
        self.glue_context.purge_s3_path(s3_storage_path, {"retentionPeriod": 0})
        self.glue_context.write_dynamic_frame.from_options(
            frame=input_dyf,
            connection_type="s3",
            connection_options={
                "path": s3_storage_path
                # "partitionKeys": ["year", "month", "day"],
            },
            format="parquet"
        )

        # Set up our SageWorks metadata (description, tags, etc)
        description = f"SageWorks data source: {self.output_uuid}"
        sageworks_meta = {"sageworks_tags": ":".join(tags)}
        for key, value in self.output_meta.items():
            sageworks_meta[key] = value

        # Create a new table in the AWS Data Catalog
        self.log.info(f"Creating Data Catalog Table: {self.output_uuid}...")
        table_input = {
            "Name": self.output_uuid,
            "Description": description,
            "Parameters": sageworks_meta,
            "TableType": "EXTERNAL_TABLE",
            "StorageDescriptor": {
                "Columns": [{"Name": col.name, "Type": col.dataType.typeName()} for col in input_dyf.schema().fields],
                "Location": s3_storage_path,
                "InputFormat": "org.apache.hadoop.mapred.TextInputFormat",
                "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
                "Compressed": True,
                "NumberOfBuckets": -1,
                "SerdeInfo": {
                    "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
                    "Parameters": {"serialization.format": "1"},
                },
            },
        }

        # Delete the Data Catalog Table if it already exists
        if overwrite:
            glue_client = boto3.client("glue")
            try:
                glue_client.delete_table(DatabaseName="sageworks", Name=self.output_uuid)
                self.log.info(f"Deleting Data Catalog Table: {self.output_uuid}...")
            except ClientError as e:
                if e.response["Error"]["Code"] != "EntityNotFoundException":
                    raise e

        self.log.info(f"Creating Data Catalog Table: {self.output_uuid}...")
        glue_client.create_table(DatabaseName="sageworks", TableInput=table_input)

        # All done!
        self.log.info(f"{self.input_uuid} --> {self.output_uuid} complete!")


if __name__ == "__main__":
    """Glue Job for the S3HeavyToDataSource Class"""

    # Get the arguments for this Glue Job
    args = getResolvedOptions(sys.argv, ["JOB_NAME"])

    # Grab the SparkContext, GlueContext, and Job
    # These are all AWS Glue Job specific
    sc = SparkContext()
    glueContext = GlueContext(sc)
    job = Job(glueContext)
    job.init(args["JOB_NAME"], args)

    # Test the Heavy Data Loader
    input_path = "s3://scp-sageworks-artifacts/incoming-data/jsonl-data/"
    data_output_uuid = "heavy_data_test"
    my_loader = S3HeavyToDataSource(glueContext, input_path, data_output_uuid)

    # Store this data as a SageWorks DataSource
    my_loader.transform()

    # Commit the Glue Job
    job.commit()
