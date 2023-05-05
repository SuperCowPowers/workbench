"""S3HeavyToDataSource: Class to move HEAVY S3 Files into a SageWorks DataSource"""
import sys
import boto3
from botocore.exceptions import ClientError

# Import all the AWS Glue Python Libraries
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.transforms import ApplyMapping


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

    @staticmethod
    def column_type_conversion(input_dyf: DynamicFrame) -> DynamicFrame:
        """Convert the column types from the input data (Spark) to the output data (Athena/Parquet)"""
        # Define the mapping from Spark data types to Athena data types
        type_mapping = {"struct": "string", "choice": "long"}

        # Get the column names and types from the input data
        column_names_types = [(col.name, col.dataType.typeName()) for col in input_dyf.schema().fields]
        input_dyf.printSchema()

        # Define the mapping from input column names/types (Spark) to output
        # column names and types that are compatible with Athena/Parquet.
        mapping = []
        for name, col_type in column_names_types:
            output_type = type_mapping.get(col_type, col_type)
            if col_type == "timestamp":
                # Apply to_timestamp transformation for timestamp columns
                output_type = f"to_timestamp({name})"
            mapping.append((name, col_type, name, output_type))
        print(mapping)

        # Apply the mapping and convert data types
        output_dyf = ApplyMapping.apply(frame=input_dyf, mappings=mapping, transformation_ctx="applymapping")
        output_dyf.printSchema()
        return output_dyf

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
        )
        self.log.info("Incoming DataFrame...")
        input_dyf.show(5)

        # Convert the columns types from Spark types to Athena/Parquet types
        output_dyf = self.column_type_conversion(input_dyf)

        # Write Parquet files to S3
        self.log.info(f"Writing Parquet files to {s3_storage_path}...")
        self.glue_context.purge_s3_path(s3_storage_path, {"retentionPeriod": 0})
        self.glue_context.write_dynamic_frame.from_options(
            frame=output_dyf,
            connection_type="s3",
            connection_options={
                "path": s3_storage_path
                # "partitionKeys": ["year", "month", "day"],
            },
            format="parquet",
        )

        # Set up our SageWorks metadata (description, tags, etc)
        description = f"SageWorks data source: {self.output_uuid}"
        sageworks_meta = {"sageworks_tags": ":".join(tags)}
        for key, value in self.output_meta.items():
            sageworks_meta[key] = value

        # Create a new table in the AWS Data Catalog
        self.log.info(f"Creating Data Catalog Table: {self.output_uuid}...")

        # Converting the Spark Types to Athena Types
        def to_athena_type(col):
            athena_type_map = {"long": "bigint", "struct": "string"}
            spark_type = col.dataType.typeName()
            return athena_type_map.get(spark_type, spark_type)

        column_name_types = [{"Name": col.name, "Type": to_athena_type(col)} for col in output_dyf.schema().fields]
        table_input = {
            "Name": self.output_uuid,
            "Description": description,
            "Parameters": sageworks_meta,
            "TableType": "EXTERNAL_TABLE",
            "StorageDescriptor": {
                "Columns": column_name_types,
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
    input_path = "s3://scp-sageworks-artifacts/incoming-data/dns/"
    data_output_uuid = "heavy_dns"
    my_loader = S3HeavyToDataSource(glueContext, input_path, data_output_uuid)

    # Store this data as a SageWorks DataSource
    my_loader.transform()

    # Commit the Glue Job
    job.commit()
