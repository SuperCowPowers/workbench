"""Helper functions for working with ISO-8601 formatted dates and times"""

from datetime import datetime, timezone


def datetime_to_iso8601(dt):
    """Convert datetime to string in UTC format (yyyy-MM-dd'T'HH:mm:ss.SSSZ)
    Note: This particular format is required by AWS Feature Store"""

    # Check for TimeZone
    if dt.tzinfo is None:
        dt = dt.tz_localize(timezone.utc)

    # Convert to ISO-8601 String
    iso_str = dt.astimezone(timezone.utc).isoformat("T", "milliseconds")
    return iso_str.replace("+00:00", "Z")


def iso8601_to_datetime(iso8601_str):
    """Convert ISO-8601 string to datetime object"""
    if "Z" in iso8601_str:
        iso8601_str = iso8601_str.replace("Z", "+00:00")
    return datetime.fromisoformat(iso8601_str).replace(tzinfo=timezone.utc)


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
