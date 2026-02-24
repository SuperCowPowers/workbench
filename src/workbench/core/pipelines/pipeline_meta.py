"""PipelineMeta: Resolves pipeline metadata from environment configuration."""

import json
import logging
import os

# Sentinel for distinguishing "no default provided" from "default=None"
_MISSING = object()


class PipelineMeta:
    """PipelineMeta: Resolves pipeline metadata from the PIPELINE_META environment variable.

    Reads pipeline configuration from the PIPELINE_META environment variable (JSON dict).
    Raises RuntimeError if PIPELINE_META is not set or contains invalid JSON.

    Common Usage:
        ```python
        from workbench.core.pipelines.pipeline_meta import PipelineMeta

        pm = PipelineMeta()
        model_name = pm.model_name
        endpoint_name = pm.endpoint_name
        mode = pm.mode
        serverless = pm.serverless

        # Access arbitrary keys (fails hard if key missing and no default)
        custom_value = pm.get("custom_key", default="fallback")
        ```

    Environment Variable:
        PIPELINE_META: JSON dict with pipeline configuration, e.g.:
        ```
        PIPELINE_META='{"mode": "dt", "model_name": "my-model-dt",
                       "endpoint_name": "my-endpoint-dt", "serverless": true}'
        ```
    """

    def __init__(self):
        """Initialize PipelineMeta from the PIPELINE_META environment variable."""
        self.log = logging.getLogger("workbench")
        self._meta = {}
        self._owner = "test"
        self._resolve()

    def get(self, key: str, default=_MISSING):
        """Get a value from the pipeline metadata.

        Args:
            key (str): The key to look up
            default: Default value if key is not found (raises RuntimeError if omitted)

        Returns:
            The value for the key, or default if not found
        """
        if key in self._meta:
            return self._meta[key]
        if default is not _MISSING:
            return default
        msg = f"PipelineMeta: Key '{key}' not found in PIPELINE_META"
        self.log.critical(msg)
        raise RuntimeError(msg)

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
        """The pipeline execution mode (e.g., 'dt', 'ts', 'promote', 'test_promote')."""
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
            - dt / ts: "DT"
            - promote: "Pro-{owner}"
            - test_promote: "Pro-Test-{owner}"
            - any other: "{owner}"

        Returns:
            The resolved owner string
        """
        mode = self.mode
        owner = self._owner
        if mode in ("dt", "ts"):
            return "DT"
        elif mode == "promote":
            return f"Pro-{owner}"
        elif mode == "test_promote":
            return f"Pro-Test-{owner}"
        else:
            return owner

    def _resolve(self):
        """Resolve pipeline metadata from the PIPELINE_META environment variable."""
        pipeline_meta_json = os.environ.get("PIPELINE_META")
        if not pipeline_meta_json:
            msg = "PipelineMeta: PIPELINE_META environment variable not set"
            self.log.critical(msg)
            raise RuntimeError(msg)
        self._resolve_from_env(pipeline_meta_json)

    def _resolve_from_env(self, pipeline_meta_json: str):
        """Parse pipeline metadata from the PIPELINE_META environment variable.

        Args:
            pipeline_meta_json (str): JSON string from PIPELINE_META env var
        """
        try:
            self._meta = json.loads(pipeline_meta_json)
        except json.JSONDecodeError as e:
            msg = f"PipelineMeta: Failed to parse PIPELINE_META: {e}"
            self.log.critical(msg)
            raise RuntimeError(msg)

        # Sensible defaults for mode and serverless (launcher always provides these)
        self._meta.setdefault("mode", "dt")
        self._meta.setdefault("serverless", True)
        self.log.info(f"PipelineMeta: mode={self._meta['mode']}, model={self._meta.get('model_name', 'N/A')}")

    def __repr__(self) -> str:
        """String representation of this PipelineMeta."""
        return (
            f"PipelineMeta(mode={self._meta.get('mode')}, model={self._meta.get('model_name')}, "
            f"endpoint={self._meta.get('endpoint_name')}, serverless={self._meta.get('serverless')})"
        )


if __name__ == "__main__":
    """Exercise the PipelineMeta class"""

    # Set up PIPELINE_META env var
    os.environ["PIPELINE_META"] = json.dumps(
        {
            "mode": "dt",
            "model_name": "ppb-human-free-reg-xgb-1-dt",
            "endpoint_name": "ppb-human-free-reg-xgb-1-dt",
            "serverless": True,
        }
    )
    pm = PipelineMeta()
    pm.set_owner("BW")
    print(f"PipelineMeta: {pm}")
    print(f"  model_name: {pm.model_name}")
    print(f"  endpoint_name: {pm.endpoint_name}")
    print(f"  mode: {pm.mode}")
    print(f"  serverless: {pm.serverless}")
    print(f"  owner: {pm.dynamic_owner()}")
    print(f"  custom key: {pm.get('custom_key', 'not set')}")
