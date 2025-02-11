"""JSON Utilities"""

import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime, date

# Local Imports
from workbench.utils.datetime_utils import datetime_to_iso8601, iso8601_to_datetime

log = logging.getLogger("workbench")


# Custom JSON Encoder (see matched decoder below)
class CustomEncoder(json.JSONEncoder):
    def default(self, obj) -> object:
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
                data = {"__dataframe__": True, "df": obj.to_dict()}
                data["index"] = obj.index.tolist()
                data["index_name"] = obj.index.name
                return data
            else:
                return super(CustomEncoder, self).default(obj)
        except Exception as e:
            log.error(f"Failed to encode object: {e}")
            return super(CustomEncoder, self).default(obj)


# Custom JSON Decoder (see matched encoder above)
def custom_decoder(dct):
    try:
        if "__datetime__" in dct:
            return iso8601_to_datetime(dct["datetime"])
        elif "__dataframe__" in dct:
            df = pd.DataFrame.from_dict(dct["df"])
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

    # Test DataFrame with named index
    print("\nTesting DataFrame with named index:")
    df_with_index = pd.DataFrame({"A": [1, 2, 3]})
    df_with_index.index.name = "index_name"
    encoded_df = json.dumps(df_with_index, cls=CustomEncoder)
    decoded_df = json.loads(encoded_df, object_hook=custom_decoder)

    print("Original DataFrame index name:", df_with_index.index.name)
    print("Decoded DataFrame index name:", decoded_df.index.name)  # Likely None
