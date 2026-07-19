"""Dashboard URL helpers.

The dashboard base URL comes from the `DASHBOARD_URL` config value. It is not
discoverable from AWS: the CDK stack only records the load balancer's DNS name,
and deployments usually sit behind a friendly domain that nothing in the stack
knows about.
"""

from typing import Optional
from urllib.parse import quote

# Workbench Imports
from workbench.utils.config_manager import ConfigManager

# Artifact type -> dashboard page
PAGE_FOR_TYPE = {
    "data_source": "data_sources",
    "feature_set": "feature_sets",
    "model": "models",
    "endpoint": "endpoints",
    "pipeline": "ml_pipelines",
}


def dashboard_url() -> Optional[str]:
    """The configured dashboard base URL, or None if not set.

    Returns:
        Optional[str]: Base URL without a trailing slash.
    """
    url = ConfigManager().get_config("DASHBOARD_URL")
    return url.rstrip("/") if url else None


def artifact_url(artifact_type: str, name: str) -> Optional[str]:
    """Build a deep link to an artifact's dashboard page.

    Args:
        artifact_type (str): One of data_source, feature_set, model, endpoint, pipeline.
        name (str): The artifact name.

    Returns:
        Optional[str]: The URL, or None if DASHBOARD_URL is unset or the type is unknown.
    """
    base = dashboard_url()
    page = PAGE_FOR_TYPE.get(artifact_type)
    if not base or not page:
        return None
    return f"{base}/{page}?name={quote(name, safe='')}"
