"""Welcome to the Workbench DataLoaders Heavy Classes

These classes provide low-level APIs for loading larger data into AWS services

- S3HeavyToDataSource: Loads large data from S3 into a DataSource
"""

from .s3_heavy_to_data_source import S3HeavyToDataSource

__all__ = ["S3HeavyToDataSource"]
