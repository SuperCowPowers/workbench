"""Timing / throughput profile for the async 3D descriptor endpoint.

Long-marked: runs a representative drug-like set through smiles-to-3d-full-v1,
collects per-molecule ``desc3d_compute_time_s``, and projects whether a
MetaEndpoint batch of ``META_BATCH`` rows clears SageMaker async's 60-minute
per-invocation cap on a ``WORKERS``-instance child fleet.

This is the data we use to decide whether the conformer tier (and/or the meta
batch size) needs lowering — see the throughput model in the test output. It is
informational by design: the hard assertions only guard against the endpoint
being broken or absurdly slow; the projection is printed for human judgment
because wall time is instance- and load-dependent.

Reference set:
    s3://workbench-public-data/comp_chem/reference_compounds/3d_perf
"""

import pandas as pd
import pytest

from workbench.api import Endpoint, PublicData

pytestmark = pytest.mark.long

ENDPOINT_NAME = "smiles-to-3d-full-v1"
REFERENCE_DATASET = "comp_chem/reference_compounds/3d_perf"

# Irganox 1010 (CAS 6683-19-8) — the public skip:cost surrogate. Included so the
# profile proves the cost guard rejects a pathological molecule cheaply on the
# live endpoint instead of burning ~40 min of xTB on it.
IRGANOX_ID = 999999
IRGANOX_1010 = (
    "O=C(OCC(COC(=O)CCc1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1)(COC(=O)CCc1cc(C(C)(C)C)"
    "c(O)c(C(C)(C)C)c1)COC(=O)CCc1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1)CCc1cc(C(C)(C)C)"
    "c(O)c(C(C)(C)C)c1"
)

# Throughput model parameters (mirror the deployed config).
WORKERS = 8  # child instances processing concurrently
CHILD_BATCH = 5  # rows per child invocation
META_BATCH = 200  # rows a single meta invocation must finish under the cap
ASYNC_CAP_S = 3600  # SageMaker async per-invocation ceiling (60 min)

# Median drug-like molecule should be well under this; tripping it means the
# endpoint is pathologically slow, not just "heavier than expected".
MAX_REASONABLE_MEDIAN_S = 300.0


@pytest.fixture(scope="module")
def result_df():
    ref = PublicData().get(REFERENCE_DATASET)
    assert ref is not None, f"Reference dataset not found: {REFERENCE_DATASET}"
    # Append the Irganox skip:cost surrogate to the drug-like set.
    payload = pd.concat(
        [ref[["id", "smiles"]], pd.DataFrame({"id": [IRGANOX_ID], "smiles": [IRGANOX_1010]})],
        ignore_index=True,
    )
    pred = Endpoint(ENDPOINT_NAME).inference(payload)
    names = dict(zip(ref["id"], ref["name"]))
    names[IRGANOX_ID] = "irganox_1010"
    pred["name"] = pred["id"].map(names)
    return pred


def test_xtb_is_active(result_df):
    """Sanity: the timing we're profiling is actually the GFN2-xTB path."""
    ok = result_df[result_df["desc3d_status"] == "ok"]
    assert not ok.empty, "no compounds computed successfully"
    methods = set(ok["desc3d_energy_method"].unique())
    assert methods == {"GFN2-xTB"}, f"expected GFN2-xTB weighting, got {methods}"


def test_cost_guard_skips_pathological(result_df):
    """The Irganox surrogate is cheaply rejected (skip:cost), not computed.

    This is the molecule that would otherwise blow the time budget — proving the
    guard fires on the live endpoint is the whole point of including it here.
    """
    irganox = result_df[result_df["id"] == IRGANOX_ID]
    assert not irganox.empty, "Irganox surrogate missing from results"
    status = irganox.iloc[0]["desc3d_status"]
    assert status == "skip:cost", f"expected Irganox to skip:cost, got {status!r}"


def test_timing_profile_and_throughput(result_df):
    """Profile per-molecule xTB time and project the meta-batch wall time."""
    ok = result_df[result_df["desc3d_status"] == "ok"].copy()
    t = ok["desc3d_compute_time_s"].astype(float)
    assert (t > 0).all(), "compute times must be positive"

    mean_s, median_s, max_s = t.mean(), t.median(), t.max()

    # Balanced-fleet model: a B-row meta invocation does B molecules of xTB work
    # spread over WORKERS instances → wall ≈ B * per_mol / WORKERS.
    def project(per_mol_s: float) -> float:
        return META_BATCH * per_mol_s / WORKERS

    proj_mean = project(mean_s)
    proj_worst = project(max_s)

    print(f"\n--- per-molecule desc3d_compute_time_s (n={len(t)}) ---")
    print(f"  mean={mean_s:.1f}s  median={median_s:.1f}s  max={max_s:.1f}s")
    print("  slowest:")
    slow = ok.nlargest(5, "desc3d_compute_time_s")[["name", "desc3d_conf_count", "desc3d_compute_time_s"]]
    print(slow.to_string(index=False))

    print(f"\n--- throughput projection (B={META_BATCH} rows, W={WORKERS}, cap={ASYNC_CAP_S}s) ---")
    print(f"  mean-mix  : {proj_mean / 60:.1f} min  ({proj_mean / ASYNC_CAP_S:.0%} of cap)")
    print(f"  worst-mix : {proj_worst / 60:.1f} min  ({proj_worst / ASYNC_CAP_S:.0%} of cap)")
    # Largest meta batch that clears the cap at the mean mix.
    safe_batch = int(ASYNC_CAP_S * WORKERS / mean_s) if mean_s else 0
    print(f"  max safe meta batch @ mean mix: ~{safe_batch} rows")
    if proj_mean > ASYNC_CAP_S:
        print("  ⚠️  mean-mix projection EXCEEDS cap — lower the tier or meta batch")

    # Hard assertion: only fail on pathological slowness, not normal heaviness.
    assert (
        median_s < MAX_REASONABLE_MEDIAN_S
    ), f"median per-molecule time {median_s:.0f}s > {MAX_REASONABLE_MEDIAN_S:.0f}s — endpoint is too slow"


if __name__ == "__main__":
    ep = Endpoint(ENDPOINT_NAME)
    ref = PublicData().get(REFERENCE_DATASET)
    res = ep.inference(ref[["id", "smiles"]].copy()).merge(ref[["id", "name"]], on="id", how="left")
    test_xtb_is_active(res)
    test_timing_profile_and_throughput(res)
    print("\nTiming profile complete.")
