"""Re-export: :class:`UQModelV2` from :mod:`workbench.algorithms.dataframe.uq_model_v2`.

Part of the :mod:`workbench.endpoints` contract — model scripts import their
endpoint-safe surface exclusively from ``workbench.endpoints.*`` so the
endpoint-import-smoke CI job can enforce the lightweight install boundary.
"""

from workbench.algorithms.dataframe.uq_model_v2 import UQModelV2  # noqa: F401

__all__ = ["UQModelV2"]
