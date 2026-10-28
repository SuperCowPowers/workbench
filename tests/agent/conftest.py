"""Keep Bosco tests off the user's real session store.

Every Bosco turn rewrites the last-session record, so any test that drives `_ask`
would otherwise write to the developer's own store (local disk or AWS).
"""

import pytest

from workbench.agent import bosco as bosco_mod
from workbench.local import ParameterStore as LocalParameterStore
from workbench.utils import bosco_utils


@pytest.fixture(autouse=True)
def isolated_session_store(tmp_path, monkeypatch):
    """A throwaway local store, with this process's cached previous session cleared."""
    store = LocalParameterStore()
    store.root = str(tmp_path / "parameter_store")
    monkeypatch.setattr(bosco_utils, "param_store", lambda: store)
    monkeypatch.setattr(bosco_mod, "_exchanges", [])
    bosco_utils.previous_session.cache_clear()
    bosco_mod._previous_hint.cache_clear()
    yield store
    bosco_utils.previous_session.cache_clear()
    bosco_mod._previous_hint.cache_clear()
