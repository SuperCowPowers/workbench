"""Unit tests for AsyncEndpointCore cold-start warm-up.

Local + fast (no AWS). Exercises the warm loop on a bare instance with its
AWS-touching helpers stubbed, so we test the control flow: always-fire then return
when warm, short-circuit when serverless, fire-then-wait when cold, fail-fast when
the warmer can't be queued, and a clean retryable error when it never comes up.
"""

from workbench.core.artifacts import async_endpoint_core as mod
from workbench.core.artifacts.async_endpoint_core import AsyncEndpointCore
from workbench.utils.async_endpoint_utils import EndpointWarmingError


class _Clock:
    """Fake clock so the warm loop doesn't wait in real time."""

    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _bare_core():
    c = object.__new__(AsyncEndpointCore)  # skip __init__ (no AWS)
    c.name = "ep"
    c.endpoint_name = "ep"
    c.is_serverless = lambda: False
    return c


def test_warm_self_always_fires_then_returns_when_warm():
    c = _bare_core()
    c._current_instances = lambda: 2  # already serving (or fresh-deploy transient)
    fired = []
    c._fire_warmer = lambda: fired.append(1) or None
    c._warm_self()
    assert fired == [1]  # always fires the warmer, even when instances already appear up
    # (returns immediately — the poll checks before sleeping, so no fake clock needed)


def test_warm_self_noop_when_serverless():
    c = _bare_core()
    c.is_serverless = lambda: True
    polled = []
    c._current_instances = lambda: polled.append(1) or 0
    c._fire_warmer = lambda: polled.append("fire")
    c._warm_self()
    assert polled == []  # serverless → return before any poll or warmer


def test_warm_self_fires_then_returns_when_instance_appears(monkeypatch):
    monkeypatch.setattr(mod, "time", _Clock())
    c = _bare_core()
    seq = [0, 0, 1]  # cold, cold, then one instance up
    state = {"i": 0}

    def current():
        v = seq[min(state["i"], len(seq) - 1)]
        state["i"] += 1
        return v

    c._current_instances = current
    fired = []
    c._fire_warmer = lambda: fired.append(1) or None  # None == queued OK
    c._warm_self()
    assert fired == [1]  # fired exactly one warmer, then polled to ready


def test_warm_self_fails_fast_when_warmer_cannot_queue():
    c = _bare_core()
    c._current_instances = lambda: 0
    c._fire_warmer = lambda: "AccessDenied: s3:PutObject"  # could not queue
    raised = None
    try:
        c._warm_self()
    except EndpointWarmingError as e:
        raised = str(e)
    assert raised is not None and "AccessDenied" in raised  # surfaces the real cause, no 15-min stall


def test_warm_self_raises_when_never_warms(monkeypatch):
    monkeypatch.setattr(mod, "time", _Clock())
    c = _bare_core()
    c._current_instances = lambda: 0  # scaling, never serves
    c._fire_warmer = lambda: None
    c._live_instance_counts = lambda: {"current": 0, "desired": 1}
    raised = None
    try:
        c._warm_self()
    except EndpointWarmingError as e:
        raised = str(e)
    assert raised is not None and "retry shortly" in raised
