"""Navigation helpers for the models dashboard page."""

from urllib.parse import parse_qs, urlencode, urlparse


def selected_model_url(selected_rows, href):
    """Return the canonical model page URL for the current table selection."""
    if not selected_rows or selected_rows[0] is None:
        return None

    model_name = selected_rows[0].get("name")
    if not model_name:
        return None

    parsed = urlparse(href or "/models")
    if parsed.path != "/models":
        return None

    current_name = parse_qs(parsed.query).get("name", [None])[0]
    if current_name == model_name:
        return None

    return f"/models?{urlencode({'name': model_name})}"
