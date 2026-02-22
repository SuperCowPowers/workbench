"""PipelineMeta: Resolves pipeline metadata from environment configuration."""

import json
import logging
import os
from datetime import datetime


class PipelineMeta:
    """PipelineMeta: Resolves pipeline metadata from the PIPELINE_META environment variable.

    Reads pipeline configuration from the PIPELINE_META environment variable (JSON dict).
    If not set, provides sensible defaults for local dev/testing.

    Common Usage:
        ```python
        from workbench.core.pipelines.pipeline_meta import PipelineMeta

        pm = PipelineMeta()
        model_name = pm.model_name
        endpoint_name = pm.endpoint_name
        mode = pm.mode
        serverless = pm.serverless

        # Access arbitrary keys
        custom_value = pm.get("custom_key", default="fallback")
        ```

    Environment Variable:
        PIPELINE_META: JSON dict with pipeline configuration, e.g.:
        ```
        PIPELINE_META='{"mode": "dt", "model_name": "my-model-dt", "endpoint_name": "my-endpoint-dt", "serverless": true}'
        ```
    """

    def __init__(self):
        """Initialize PipelineMeta from the PIPELINE_META environment variable."""
        self.log = logging.getLogger("workbench")
        self._meta = {}
        self._owner = None
        self._resolve()

    def get(self, key: str, default=None):
        """Get a value from the pipeline metadata.

        Args:
            key (str): The key to look up
            default: Default value if key is not found

        Returns:
            The value for the key, or default if not found
        """
        return self._meta.get(key, default)

    @property
    def model_name(self) -> str:
        """The resolved model name."""
        return self._meta["model_name"]

    @property
    def endpoint_name(self) -> str:
        """The resolved endpoint name."""
        return self._meta["endpoint_name"]

    @property
    def mode(self) -> str:
        """The pipeline execution mode (e.g., 'dt', 'promote', 'dev')."""
        return self._meta["mode"]

    @property
    def serverless(self) -> bool:
        """Whether to use serverless inference."""
        return self._meta["serverless"]

    def set_owner(self, owner: str):
        """Set the owner for dynamic owner resolution.

        Args:
            owner (str): The owner identifier (e.g., "BW", "Bob")
        """
        self._owner = owner

    def dynamic_owner(self) -> str:
        """Return mode-appropriate owner string.

        Uses the owner set via set_owner() and transforms based on mode:
            - dt / temporal_split: "DT"
            - promote: "Pro-{owner}"
            - test_promote: "Pro-Test-{owner}"
            - dev (or any other): "{owner}"

        Returns:
            The resolved owner string
        """
        if self._owner is None:
            return "test"
        mode = self.mode
        owner = self._owner
        if mode in ("dt", "temporal_split"):
            return "DT"
        elif mode == "promote":
            return f"Pro-{owner}"
        elif mode == "test_promote":
            return f"Pro-Test-{owner}"
        else:
            return owner

    def _resolve(self):
        """Resolve pipeline metadata from environment or defaults."""
        pipeline_meta_json = os.environ.get("PIPELINE_META")
        if pipeline_meta_json:
            self._resolve_from_env(pipeline_meta_json)
        else:
            self._resolve_defaults()

    def _resolve_from_env(self, pipeline_meta_json: str):
        """Parse pipeline metadata from the PIPELINE_META environment variable.

        Args:
            pipeline_meta_json (str): JSON string from PIPELINE_META env var
        """
        try:
            self._meta = json.loads(pipeline_meta_json)
        except json.JSONDecodeError as e:
            self.log.error(f"Failed to parse PIPELINE_META: {e}")
            self.log.warning("Falling back to defaults")
            self._resolve_defaults()
            return

        # Ensure required keys have defaults
        self._meta.setdefault("mode", "dev")
        self._meta.setdefault("serverless", True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        self._meta.setdefault("model_name", f"test-{timestamp}")
        self._meta.setdefault("endpoint_name", f"test-{timestamp}")
        self.log.info(f"PipelineMeta: mode={self._meta['mode']}, model={self._meta['model_name']}")

    def _resolve_defaults(self):
        """Set default metadata for local dev/testing."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        self._meta = {
            "mode": "dev",
            "model_name": f"test-{timestamp}",
            "endpoint_name": f"test-{timestamp}",
            "serverless": True,
        }
        self.log.info(f"PipelineMeta: No PIPELINE_META env var, using defaults ({self._meta['model_name']})")

    def __repr__(self) -> str:
        """String representation of this PipelineMeta."""
        return (
            f"PipelineMeta(mode={self._meta.get('mode')}, model={self._meta.get('model_name')}, "
            f"endpoint={self._meta.get('endpoint_name')}, serverless={self._meta.get('serverless')})"
        )


if __name__ == "__main__":
    """Exercise the PipelineMeta class"""

    # Test with no env var (defaults)
    pm = PipelineMeta()
    print(f"Default: {pm}")
    print(f"  model_name: {pm.model_name}")
    print(f"  endpoint_name: {pm.endpoint_name}")
    print(f"  mode: {pm.mode}")
    print(f"  serverless: {pm.serverless}")

    # Test with PIPELINE_META env var
    os.environ["PIPELINE_META"] = json.dumps({
        "mode": "dt",
        "model_name": "ppb-human-free-reg-xgb-1-dt",
        "endpoint_name": "ppb-human-free-reg-xgb-1-dt",
        "serverless": True,
    })
    pm2 = PipelineMeta()
    print(f"\nWith env var: {pm2}")
    print(f"  model_name: {pm2.model_name}")
    print(f"  mode: {pm2.mode}")
    print(f"  custom key: {pm2.get('custom_key', 'not set')}")
