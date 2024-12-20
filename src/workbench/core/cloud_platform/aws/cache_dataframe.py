"""Cache Dataframe Decorator: Easy decorator to cache DataFrames using AWS S3/Parquet/Snappy"""

import logging
from functools import wraps

# Set up logging
log = logging.getLogger("workbench")


def cache_dataframe(location: str):
    """Decorator to cache DataFrame results in DFStore at a location based on `self.uuid`.

    Args:
        location (str): The final part of the cache path (e.g., 'sample', 'features').

    This decorator assumes it is applied to a Workbench Artifact class (has self.uuid and self.df_cache).
    """

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # Construct the full cache location
            df_path = f"{self.uuid}/{location}".replace("//", "/")
            full_path = f"{self.df_cache.path_prefix}/{df_path}"

            # Check for cached data at the specified location
            cached_df = self.df_cache.get(df_path)
            if cached_df is not None:
                log.info(f"Returning cached DataFrame from {full_path}")
                return cached_df

            # Call the original method to fetch the DataFrame
            dataframe = method(self, *args, **kwargs)

            # Cache the result at the specified location
            log.info(f"Caching DataFrame to {full_path}")
            self.df_cache.upsert(df_path, dataframe)
            return dataframe

        return wrapper

    return decorator


if __name__ == "__main__":
    """Exercise the DataFrame Decorator"""

    from workbench.api.data_source import DataSource

    ds = DataSource("test_data")
    ds.sample()  # Since the method uses the decorator, the result will be cached
