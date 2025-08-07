"""JSON Utilities"""

import json
from io import StringIO
import numpy as np
import pandas as pd
import logging
from datetime import datetime, date

# Local Imports
from workbench.utils.datetime_utils import datetime_to_iso8601, iso8601_to_datetime

log = logging.getLogger("workbench")


# Custom JSON Encoder with optional precision reduction (see matched decoder below)
class CustomEncoder(json.JSONEncoder):
    def __init__(self, precision=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision = precision

    def default(self, obj):
        try:
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (datetime, date)):
                return {"__datetime__": True, "datetime": datetime_to_iso8601(obj)}
            elif isinstance(obj, pd.DataFrame):
                return {
                    "__dataframe__": True,
                    "df": obj.to_json(orient="table"),
                }
            return super().default(obj)
        except Exception as e:
            log.error(f"Failed to encode object: {e}")
            return super().default(obj)

    def encode(self, obj):
        return super().encode(self._reduce_precision(obj) if self.precision else obj)

    def _reduce_precision(self, obj):
        if isinstance(obj, float):
            return round(obj, self.precision)
        elif isinstance(obj, dict):
            return {k: self._reduce_precision(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            result = [self._reduce_precision(item) for item in obj]
            return tuple(result) if isinstance(obj, tuple) else result
        return obj


# Custom JSON Decoder (see matched encoder above)
def custom_decoder(dct):
    try:
        if "__datetime__" in dct:
            return iso8601_to_datetime(dct["datetime"])
        elif "__dataframe__" in dct:
            df_data = dct["df"]
            if isinstance(df_data, str):
                df = pd.read_json(StringIO(df_data), orient="table")
            else:
                # Old format compatibility
                log.warning("Decoding old dataframe format...")
                df = pd.DataFrame.from_dict(df_data)
                if "index" in dct:
                    df.index = dct["index"]
                    df.index.name = dct.get("index_name")
            return df
        return dct
    except Exception as e:
        log.error(f"Failed to decode object: {e}")
        return dct


if __name__ == "__main__":
    # Test the custom encoder and decoder
    test_dict = {
        "int": 42,
        "float": 3.14,
        "bool": True,
        "np_int": np.int64(42),
        "np_float": np.float64(3.14),
        "np_bool": np.bool_(True),
        "np_array": np.array([1, 2, 3]),
        "datetime": datetime.now(),
        "date": date.today(),
        "dataframe": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
        "list": [1, 2, 3],
    }

    # Encode the test dictionary
    encoded = json.dumps(test_dict, cls=CustomEncoder)
    print("Encoded JSON:")
    print(encoded)

    # Decode the encoded JSON
    decoded = json.loads(encoded, object_hook=custom_decoder)
    print("\nDecoded Dictionary:")
    print(decoded)
    print("\nDecoded DataFrame:")
    print(decoded["dataframe"])
    print("\nDecoded DataFrame Columns:")
    print(decoded["dataframe"].columns)

    # Test the precision reduction
    print("\nTesting precision reduction:")
    precision_test_dict = {
        "float": 3.141592653589793,
        "np_float": np.float64(2.718281828459045),
    }
    encoded_precision = json.dumps(precision_test_dict, cls=CustomEncoder, precision=3)
    print("Encoded with precision reduction:")
    print(encoded_precision)

    # Test DataFrame with named index
    print("\nTesting DataFrame with named index:")
    df_with_index = pd.DataFrame({"A": [1, 2, 3]})
    df_with_index.index.name = "index_name"
    encoded_df = json.dumps(df_with_index, cls=CustomEncoder)
    decoded_df = json.loads(encoded_df, object_hook=custom_decoder)

    print("Original DataFrame index name:", df_with_index.index.name)
    print("Decoded DataFrame index name:", decoded_df.index.name)

    # Dataframe Testing
    from workbench.api import DFStore

    df_store = DFStore()
    df = df_store.get("/testing/json_encoding/smart_sample_bad")
    encoded = json.dumps(df, cls=CustomEncoder)
    decoded_df = json.loads(encoded, object_hook=custom_decoder)

    # Compare original and decoded DataFrame
    from workbench.utils.pandas_utils import compare_dataframes

    compare_dataframes(df, decoded_df)
