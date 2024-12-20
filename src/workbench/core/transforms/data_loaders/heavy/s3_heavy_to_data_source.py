"""S3HeavyToDataSource: Class to move HEAVY S3 Files into a Workbench DataSource"""

import sys
import boto3
from botocore.exceptions import ClientError

# Import all the AWS Glue Python Libraries
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.transforms import Relationalize
from pyspark.sql.functions import col, to_timestamp


class S3HeavyToDataSource:
    def __init__(self, glue_context: GlueContext, input_uuid: str, output_uuid: str):
        """S3HeavyToDataSource: Class to move HEAVY S3 Files into a Workbench DataSource

        Args:
            glue_context: GlueContext, AWS Glue Specific wrapper around SparkContext
            input_uuid (str): The S3 Path to the files to be loaded
            output_uuid (str): The UUID of the Workbench DataSource to be created
        """
        self.log = glue_context.get_logger()

        # FIXME: Pull these from Parameter Store or Config
        self.input_uuid = input_uuid
        self.output_uuid = output_uuid
        self.output_meta = {"workbench_input": self.input_uuid}
        workbench_bucket = "s3://sandbox-workbench-artifacts"
        self.data_sources_s3_path = workbench_bucket + "/data-sources"

        # Our Spark Context
        self.glue_context = glue_context

    @staticmethod
    def resolve_choice_fields(dyf):
        # Get schema fields
        schema_fields = dyf.schema().fields

        # Collect choice fields
        choice_fields = [(field.name, "cast:long") for field in schema_fields if field.dataType.typeName() == "choice"]
        print(f"Choice Fields: {choice_fields}")

        # If there are choice fields, resolve them
        if choice_fields:
            dyf = dyf.resolveChoice(specs=choice_fields)

        return dyf

    def timestamp_conversions(self, dyf: DynamicFrame, time_columns: list = []) -> DynamicFrame:
        """Convert columns in the DynamicFrame to the correct data types
        Args:
            dyf (DynamicFrame): The DynamicFrame to convert
            time_columns (list): A list of column names to convert to timestamp
        Returns:
            DynamicFrame: The converted DynamicFrame
        """

        # Convert the timestamp columns to timestamp types
        spark_df = dyf.toDF()
        for column in time_columns:
            spark_df = spark_df.withColumn(column, to_timestamp(col(column)))

        # Convert the Spark DataFrame back to a Glue DynamicFrame and return
        return DynamicFrame.fromDF(spark_df, self.glue_context, "output_dyf")

    @staticmethod
    def remove_periods_from_columns(dyf: DynamicFrame) -> DynamicFrame:
        """Remove periods from column names in the DynamicFrame
        Args:
            dyf (DynamicFrame): The DynamicFrame to convert
        Returns:
            DynamicFrame: The converted DynamicFrame
        """
        # Extract the column names from the schema
        old_column_names = [field.name for field in dyf.schema().fields]

        # Create a new list of renamed column names
        new_column_names = [name.replace(".", "_") for name in old_column_names]
        print(old_column_names)
        print(new_column_names)

        # Create a new DynamicFrame with renamed columns
        for c_old, c_new in zip(old_column_names, new_column_names):
            dyf = dyf.rename_field(f"`{c_old}`", c_new)
        return dyf

    def transform(
        self,
        input_type: str = "json",
        timestamp_columns: list = None,
        output_format: str = "parquet",
    ):
        """Convert the CSV or JSON data into Parquet Format in the Workbench S3 Bucket, and
        store the information about the data to the AWS Data Catalog workbench database
        Args:
            input_type (str): The type of input files, either 'csv' or 'json'
            timestamp_columns (list): A list of column names to convert to timestamp
            output_format (str): The format of the output files, either 'parquet' or 'orc'
        """

        # Add some tags here
        tags = ["heavy"]

        # Create the Output Parquet file S3 Storage Path
        s3_storage_path = f"{self.data_sources_s3_path}/{self.output_uuid}"

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
        input_dyf.printSchema()

        # Resolve Choice fields
        resolved_dyf = self.resolve_choice_fields(input_dyf)

        # The next couple of lines of code is for un-nesting any nested JSON
        # Create a Dynamic Frame Collection (dfc)
        dfc = Relationalize.apply(resolved_dyf, name="root")

        # Aggregate the collection into a single dynamic frame
        output_dyf = dfc.select("root")

        print("Before TimeStamp Conversions")
        output_dyf.printSchema()

        # Convert any timestamp columns
        output_dyf = self.timestamp_conversions(output_dyf, timestamp_columns)

        # Relationalize will put periods in the column names. This will cause
        # problems later when we try to create a FeatureSet from this DataSource
        output_dyf = self.remove_periods_from_columns(output_dyf)

        print("After TimeStamp Conversions and Removing Periods from column names")
        output_dyf.printSchema()

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
            format=output_format,
        )

        # Set up our Workbench metadata (description, tags, etc)
        description = f"Workbench data source: {self.output_uuid}"
        workbench_meta = {"workbench_tags": self.tag_delimiter.join(tags)}
        for key, value in self.output_meta.items():
            workbench_meta[key] = value

        # Create a new table in the AWS Data Catalog
        self.log.info(f"Creating Data Catalog Table: {self.output_uuid}...")

        # Converting the Spark Types to Athena Types
        def to_athena_type(col):
            athena_type_map = {"long": "bigint"}
            spark_type = col.dataType.typeName()
            return athena_type_map.get(spark_type, spark_type)

        column_name_types = [{"Name": col.name, "Type": to_athena_type(col)} for col in output_dyf.schema().fields]

        # Our parameters for the Glue Data Catalog are different for Parquet and ORC
        if output_format == "parquet":
            glue_input_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
            glue_output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"
            serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
        else:
            glue_input_format = "org.apache.hadoop.hive.ql.io.orc.OrcInputFormat"
            glue_output_format = "org.apache.hadoop.hive.ql.io.orc.OrcInputFormat"
            serialization_library = "org.apache.hadoop.hive.ql.io.orc.OrcSerde"

        table_input = {
            "Name": self.output_uuid,
            "Description": description,
            "Parameters": workbench_meta,
            "TableType": "EXTERNAL_TABLE",
            "StorageDescriptor": {
                "Columns": column_name_types,
                "Location": s3_storage_path,
                "InputFormat": glue_input_format,
                "OutputFormat": glue_output_format,
                "Compressed": True,
                "SerdeInfo": {
                    "SerializationLibrary": serialization_library,
                },
            },
        }

        # Delete the Data Catalog Table if it already exists
        glue_client = boto3.client("glue")
        try:
            glue_client.delete_table(DatabaseName="workbench", Name=self.output_uuid)
            self.log.info(f"Deleting Data Catalog Table: {self.output_uuid}...")
        except ClientError as e:
            if e.response["Error"]["Code"] != "EntityNotFoundException":
                raise e

        self.log.info(f"Creating Data Catalog Table: {self.output_uuid}...")
        glue_client.create_table(DatabaseName="workbench", TableInput=table_input)

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
    input_path = "s3://sandbox-workbench-artifacts/incoming-data/dns/"
    data_output_uuid = "heavy_dns"
    my_loader = S3HeavyToDataSource(glueContext, input_path, data_output_uuid)

    # Store this data as a Workbench DataSource
    my_loader.transform(timestamp_columns=["timestamp"])

    # Commit the Glue Job
    job.commit()
