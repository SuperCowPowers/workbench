"""Read-side helpers for *feature endpoints* — endpoints that take an input
column (typically ``smiles``) and emit computed feature columns.

Feature endpoints register their output columns in ParameterStore via
:meth:`EndpointCore.register_features`, under the convention:

    /workbench/feature_lists/<endpoint_name>

This module exposes the path convention and a tiny lookup helper so
downstream scripts don't need an endpoint instance to discover what a
given feature endpoint produces.

Example:

    >>> from workbench.utils.feature_endpoint_utils import get_endpoint_features
    >>> get_endpoint_features("smiles-to-2d-v1")
    ['AATS0d', 'AATS0dv', ..., 'Mordred_ZMIC5']
"""

from __future__ import annotations

from typing import List, Optional

from workbench.api import ParameterStore

FEATURE_LIST_PREFIX = "/workbench/feature_lists"


def feature_list_key(endpoint_name: str) -> str:
    """Return the ParameterStore key under which an endpoint's feature
    columns are registered.

    Example:
        >>> feature_list_key("smiles-to-2d-v1")
        '/workbench/feature_lists/smiles-to-2d-v1'
    """
    return f"{FEATURE_LIST_PREFIX}/{endpoint_name}"


def get_endpoint_features(endpoint_name: str) -> Optional[List[str]]:
    """Look up the feature columns registered for a feature endpoint.

    Args:
        endpoint_name: e.g. ``"smiles-to-2d-v1"``.

    Returns:
        List of feature column names, or ``None`` if the endpoint hasn't
        registered its features yet. Call
        :meth:`Endpoint.register_features` from the endpoint's deploy
        script to populate the list.
    """
    return ParameterStore().get(feature_list_key(endpoint_name))
