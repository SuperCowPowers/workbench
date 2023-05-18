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
from awsglue.transforms import Relationalize
from pyspark.sql.functions import col, to_timestamp


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

    def column_conversions(self, dyf: DynamicFrame, time_columns: list = []) -> DynamicFrame:
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

        # Convert the Spark DataFrame back to a Glue DynamicFrame
        output_dyf = DynamicFrame.fromDF(spark_df, self.glue_context, "output_dyf")

        # Now resolve any 'choice' columns
        specs = [(field.name, "cast:long") for field in dyf.schema().fields if field.dataType.typeName() == "choice"]
        print(specs)
        if specs:
            output_dyf = output_dyf.resolveChoice(specs=specs)
        return output_dyf

    @staticmethod
    def remove_periods_from_column_names(dyf: DynamicFrame) -> DynamicFrame:
        """Remove periods from column names in the DynamicFrame
        Args:
            dyf (DynamicFrame): The DynamicFrame to convert
        Returns:
            DynamicFrame: The converted DynamicFrame
        """
        # Extract the column names from the schema
        old_column_names = [field.name for field in dyf.schema().fields]

        # Create a new list of renamed column names
        new_column_names = [name.replace('.', '_') for name in old_column_names]
        print(old_column_names)
        print(new_column_names)

        # Create a new DynamicFrame with renamed columns
        for c_old, c_new in zip(old_column_names, new_column_names):
            dyf = dyf.rename_field(f"`{c_old}`", c_new)
        return dyf

    def transform(self, input_type: str = "json", timestamp_columns: list = None):
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

        # Create a Dynamic Frame Collection (dfc)
        dfc = Relationalize.apply(input_dyf, name="root")

        # Aggregate the collection into a single dynamic frame
        all_data = dfc.select("root")

        print('Before Column Conversions')
        all_data.printSchema()

        # Relationalize will put periods in the column names. This will cause
        # problems later when we try to create a FeatureSet from this DataSource
        output_dyf = self.remove_periods_from_column_names(all_data)

        # Convert the columns types from Spark types to Athena/Parquet types
        output_dyf = self.column_conversions(output_dyf, timestamp_columns)

        print('After Column Conversions')
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
            format="orc",
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
            athena_type_map = {"long": "bigint"}
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
                "Compressed": True,
                "SerdeInfo": {
                    # "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
                    "SerializationLibrary": "org.apache.hadoop.hive.ql.io.orc.OrcSerde",
                },
            },
        }

        # Delete the Data Catalog Table if it already exists
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
    my_loader.transform(timestamp_columns=["timestamp"])

    # Commit the Glue Job
    job.commit()
