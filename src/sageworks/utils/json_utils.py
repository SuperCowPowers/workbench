"""JSON Utilities"""

import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime, date

# Local Imports
from sageworks.utils.datetime_utils import datetime_to_iso8601, iso8601_to_datetime

log = logging.getLogger("sageworks")


# Custom JSON Encoder (see matched decoder below)
class CustomEncoder(json.JSONEncoder):
    def default(self, obj) -> object:
        try:
            if isinstance(obj, dict):
                return {key: self.default(value) for key, value in obj.items()}
            elif isinstance(obj, np.integer):
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
                return {"__dataframe__": True, "df": obj.to_dict()}
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
            return pd.DataFrame.from_dict(dct["df"])
        return dct
    except Exception as e:
        log.error(f"Failed to decode object: {e}")
        return dct
