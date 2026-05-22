"""Re-export: :func:`compute_morgan_fingerprints` from
:mod:`workbench.utils.chem_utils.fingerprints`.

Part of the :mod:`workbench.endpoints` contract — see :mod:`workbench.endpoints`.
"""

from workbench.utils.chem_utils.fingerprints import compute_morgan_fingerprints  # noqa: F401

__all__ = ["compute_morgan_fingerprints"]
