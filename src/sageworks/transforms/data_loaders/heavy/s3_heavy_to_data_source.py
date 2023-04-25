"""S3HeavyToDataSource: Class to move HEAVY S3 Files into a SageWorks DataSource"""
import sys
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext

# SageWorks imports
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput


class S3HeavyToDataSource(Transform):
    def __init__(self, job_name: str, input_uuid: str, output_uuid: str):
        """S3HeavyToDataSource: Class to move HEAVY S3 Files into a SageWorks DataSource
        Args:
            job_name (str): The name of the AWS Glue Job
            input_uuid (str): The S3 Path to the files to be loaded
            output_uuid (str): The UUID of the SageWorks DataSource to be created
        """

        # Call superclass init
        super().__init__(input_uuid, output_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.S3_OBJECT
        self.output_type = TransformOutput.DATA_SOURCE

        # These are AWS Glue Job specific
        sc = SparkContext()
        self.glueContext = GlueContext(sc)
        self.job = Job(self.glueContext)
        self.job.init(job_name, [job_name, input_uuid, output_uuid])

    def transform_impl(self, overwrite: bool = True):
        """Convert the CSV data into Parquet Format in the SageWorks S3 Bucket, and
        store the information about the data to the AWS Data Catalog sageworks database"""

        # Add some tags here
        tags = ["heavy"]

        # Create the Output Parquet file S3 Storage Path
        s3_storage_path = f"{self.data_source_s3_path}/{self.output_uuid}"

        # Read JSONL files from S3 and infer schema dynamically
        self.log.info(f"Reading JSONL files from {self.input_uuid}...")
        jsonl_dyf = self.glueContext.create_dynamic_frame_from_options(
            connection_type="s3",
            connection_options={"paths": [self.input_uuid],
                                "compressionType": "gzip",
                                "recurse": "True",
                                },
            format="json",
            format_options={'jsonPath': 'auto'},
            transformation_ctx='apply_mapping'
        )

        # Write Parquet files to S3
        self.log.info(f"Writing Parquet files to {s3_storage_path}...")
        self.glueContext.write_dynamic_frame.from_options(
            frame=jsonl_dyf,
            connection_type='s3',
            connection_options={'path': s3_storage_path},
            format='parquet',
            format_options={'compression': 'snappy', 'parquetVersion': '2.0'},
            transformation_ctx='datasink'
        )

        # Set up our SageWorks metadata
        sageworks_meta = {"sageworks_tags": tags}
        for key, value in self.output_meta.items():
            sageworks_meta[key] = value

        # Create a Glue Catalog Database Table

        # Generate a Glue Catalog Table
        description = f"SageWorks data source: {self.output_uuid}"
        self.glueContext.catalog.create_table(
            database='sageworks',  # FIXME: Have this in config
            table_name=self.output_uuid,
            location=s3_storage_path,
            schema=jsonl_dyf.schema(),
            description=description,
            parameters=sageworks_meta
        )

        # Commit the Glue Job
        self.log.info(f"{self.input_uuid} --> {self.output_uuid} complete!")
        self.job.commit()


if __name__ == "__main__":
    """Glue Job for the S3HeavyToDataSource Class"""

    # Get the arguments for this Glue Job
    args = getResolvedOptions(sys.argv, ["JOB_NAME", "SRC_PATH", "OUTPUT_UUID"])

    # Create the Data Loader
    my_loader = S3HeavyToDataSource(args['JOB_NAME'], args['SRC_PATH'], args['OUTPUT_UUID'])

    # Store this data as a SageWorks DataSource
    my_loader.transform()
