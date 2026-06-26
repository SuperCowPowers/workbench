"""Unit test for the async N-in == N-out guarantee.

Local + fast (no AWS). A chunk that fails (transport or compute) must have its
input rows passed through with NaN outputs, never silently dropped — otherwise a
single slow/timed-out chunk loses rows from the final result.
"""

from unittest.mock import MagicMock

import pandas as pd

from workbench.endpoints import async_inference as ai


def test_failed_chunk_rows_preserved_with_nan(monkeypatch):
    """One failed chunk → its rows return with NaN outputs; all rows survive."""
    df = pd.DataFrame({"id": range(20), "smiles": ["CCO"] * 20})

    def fake_invoke(runtime, s3, ep, chunk_df, bucket, prefix, idem=False):
        if 10 in set(chunk_df["id"]):  # make the chunk holding ids 10..14 fail
            return None
        out = chunk_df.copy()
        out["prediction"] = 1.23
        return out

    monkeypatch.setattr(ai, "_invoke_one_async", fake_invoke)
    monkeypatch.setattr(ai, "resolve_boto_session", lambda s: MagicMock())
    monkeypatch.setattr(ai, "instance_count_str", lambda *a: "1")

    out = ai.async_inference("ep", df, batch_size=5, s3_bucket="b")

    assert len(out) == len(df)  # N in == N out
    assert sorted(out["id"]) == list(range(20))  # every row survives
    failed = out[out["prediction"].isna()]
    assert sorted(failed["id"].tolist()) == [10, 11, 12, 13, 14]  # failed chunk → NaN outputs


def test_all_chunks_failed_still_raises(monkeypatch):
    """A total outage (every chunk fails) still raises rather than returning all-NaN."""
    df = pd.DataFrame({"id": range(10), "smiles": ["CCO"] * 10})

    monkeypatch.setattr(ai, "_invoke_one_async", lambda *a, **k: None)
    monkeypatch.setattr(ai, "resolve_boto_session", lambda s: MagicMock())
    monkeypatch.setattr(ai, "instance_count_str", lambda *a: "1")

    raised = False
    try:
        ai.async_inference("ep", df, batch_size=5, s3_bucket="b")
    except RuntimeError:
        raised = True
    assert raised, "all-chunks-failed must raise, not return a frame of NaNs"
