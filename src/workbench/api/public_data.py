"""PublicData: Re-export from workbench-bridges for convenience.

This module allows importing PublicData from either location:
    from workbench.api.public_data import PublicData
    from workbench_bridges.api.public_data import PublicData
"""

from workbench_bridges.api.public_data import PublicData

__all__ = ["PublicData"]
