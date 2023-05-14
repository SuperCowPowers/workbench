"""Helper functions for working with ISO-8601 formatted dates and times"""

from datetime import datetime, timezone
import numpy as np


def datetime_to_iso8601(datetime_field: datetime) -> str:
    """Convert datetime to string in UTC format (yyyy-MM-dd'T'HH:mm:ss.SSSZ)
    Args:
        datetime_field (datetime): The datetime object to convert
    Returns:
        str: The datetime as a string in ISO-8601 format
    Note: This particular format is required by AWS Feature Store
    """

    # Check for TimeZone
    if datetime_field.tzinfo is None:
        datetime_field = datetime_field.tz_localize(timezone.utc)

    # Convert to ISO-8601 String
    iso_str = datetime_field.astimezone(timezone.utc).isoformat("T", "milliseconds")
    return iso_str.replace("+00:00", "Z")


def iso8601_to_datetime(iso8601_str: str) -> datetime:
    """Convert ISO-8601 string to datetime object
    Args:
        iso8601_str (str): The ISO-8601 string to convert
    Returns:
        datetime: The datetime object
    """
    if "Z" in iso8601_str:
        iso8601_str = iso8601_str.replace("Z", "+00:00")
    return datetime.fromisoformat(iso8601_str).replace(tzinfo=timezone.utc)


def convert_all_to_iso8601(data):
    """Convert datetime fields to ISO-8601 string
    Args:
        data (arbitrary type): The data to convert
    Returns:
        arbitrary type: The converted data
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            result[key] = convert_all_to_iso8601(value)
        return result
    elif isinstance(data, list):
        result = []
        for item in data:
            result.append(convert_all_to_iso8601(item))
        return result
    elif isinstance(data, datetime):
        return datetime_to_iso8601(data)
    elif isinstance(data, np.int64):
        return int(data)
    else:
        return data


if __name__ == "__main__":
    """Exercise the helper functions"""

    # Test the conversion to ISO-8601
    now = datetime.now(timezone.utc)
    print(now)
    now_str = datetime_to_iso8601(now)
    print(now_str)

    # Test the conversion from ISO-8601 back to datetime
    now2 = iso8601_to_datetime(now_str)
    print(now2)

    # Test the conversion of all datetime fields to ISO-8601
    data = {"a": 1, "b": "2", "c": now, "d": {"e": 3, "f": now}}
    print(data)
    data2 = convert_all_to_iso8601(data)
    print(data2)
