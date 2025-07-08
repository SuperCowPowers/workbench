"""Cache Dataframe Decorator: Decorator to cache DataFrames using AWS S3/Parquet/Snappy"""

import logging
from functools import wraps

# Set up logging
log = logging.getLogger("workbench")


# Helper function to flatten the args and kwargs into a single string
def flatten_args_kwargs(args, kwargs):
    """Flatten the args and kwargs into a single string"""
    parts = [
        "_".join(str(arg) for arg in args) if args else "",
        "_".join(f"{k}_{v}" for k, v in sorted(kwargs.items())) if kwargs else "",
    ]
    result = "_".join(p for p in parts if p)
    return f"_{result}" if result else ""


def cache_dataframe(location: str):
    """Decorator to cache DataFrame results in DFStore at a location based on `self.name`.

    Args:
        location (str): The final part of the cache path (e.g., 'sample', 'features').

    This decorator assumes it is applied to a Workbench Artifact class (has self.name and self.df_cache).
    """

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # Create a cache key based on args and kwargs
            args_hash = flatten_args_kwargs(args, kwargs)

            # Construct the full cache location with args hash
            df_path = f"{self.name}/{location}{args_hash}"
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
    df = ds.sample()  # Since the method uses the decorator, the result will be cached
    print(df)

    # Change args to test the decorator
    df = ds.sample(rows=50)
    print(df)
