"""Profile the 3D fast endpoint's compute pipeline to identify hot spots.

Measures per-stage wall-clock (embed, MMFF optimize, energy calc, descriptor
families) on 10 public drug analogs from the 3d_perf reference set. Results
are printed as a per-molecule table and an aggregate summary so we can see
which stage actually dominates end-to-end time before touching the code.

Instrumentation is additive — no monkey-patching or library modification.
The script re-implements the hot path with timing hooks around each stage,
calling the same underlying functions the library does. If numbers here
drift from the library's throughput, the instrumentation wrapper is the
culprit; the core work is identical.

Expected workflow:
    1. Run this to get a baseline timing breakdown.
    2. Apply a candidate optimization to mol_descriptors_3d.py.
    3. Re-run and compare — the stage whose time dropped is the real win.

Usage:
    python scripts/admin/profile_3d_fast.py
    python scripts/admin/profile_3d_fast.py --n-conformers 10
    python scripts/admin/profile_3d_fast.py --compounds ritonavir,maraviroc
"""

import argparse
import logging
import time
from dataclasses import dataclass, field
from typing import List

import pandas as pd
from rdkit import Chem

from workbench.api import PublicData
from workbench.utils.chem_utils.mol_descriptors_3d import (
    BOLTZMANN_FORCE_TOL,
    boltzmann_weights,
    check_complexity,
    compute_mordred_3d_descriptors,
    compute_pharmacophore_3d_descriptors,
    compute_rdkit_3d_descriptors,
    generate_conformers,
    get_conformer_energies,
)

log = logging.getLogger("workbench")
logging.basicConfig(level=logging.INFO, format="%(message)s")

REFERENCE_KEY = "comp_chem/reference_compounds/3d_perf"


@dataclass
class StageTimings:
    """Per-molecule timing breakdown (seconds)."""

    name: str
    smiles: str
    status: str = "ok"
    embed: float = 0.0
    energies: float = 0.0
    rdkit_shape: float = 0.0
    mordred: float = 0.0
    pharm: float = 0.0
    total: float = 0.0
    n_confs: int = 0
    n_in_window: int = 0
    embed_tier: int = 0
    embed_failures: int = 0
    timeout_failures: int = 0
    per_conf_mordred: List[float] = field(default_factory=list)


def profile_molecule(name: str, smiles: str, n_conformers: int) -> StageTimings:
    """Run the fast-endpoint pipeline with timing hooks around each stage."""
    t = StageTimings(name=name, smiles=smiles)
    mol_start = time.perf_counter()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        t.status = "parse_failed"
        return t

    if (skip := check_complexity(mol)) is not None:
        t.status = skip
        return t

    mol = Chem.AddHs(mol)

    # Stage 1: embed + optimize (bundled by generate_conformers)
    ts = time.perf_counter()
    mol, info = generate_conformers(
        mol,
        n_conformers=n_conformers,
        random_seed=42,
        optimize=True,
        force_tol=BOLTZMANN_FORCE_TOL,
    )
    t.embed = time.perf_counter() - ts
    t.embed_tier = info.get("embed_tier", 0)
    t.embed_failures = info.get("embed_failures", 0)
    t.timeout_failures = info.get("timeout_failures", 0)

    if mol is None or mol.GetNumConformers() == 0:
        t.status = "embed_failed"
        t.total = time.perf_counter() - mol_start
        return t
    t.n_confs = mol.GetNumConformers()

    # Stage 2: energies (separate pass over the conformer set)
    ts = time.perf_counter()
    energies = get_conformer_energies(mol)
    conf_ids, _weights = boltzmann_weights(energies)
    t.energies = time.perf_counter() - ts
    t.n_in_window = len(conf_ids)

    # Stage 3: descriptor families (per conformer in the Boltzmann window)
    for cid in conf_ids:
        ts = time.perf_counter()
        compute_rdkit_3d_descriptors(mol, cid)
        t.rdkit_shape += time.perf_counter() - ts

        ts = time.perf_counter()
        compute_mordred_3d_descriptors(mol, cid)
        dt = time.perf_counter() - ts
        t.mordred += dt
        t.per_conf_mordred.append(dt)

        ts = time.perf_counter()
        compute_pharmacophore_3d_descriptors(mol, cid)
        t.pharm += time.perf_counter() - ts

    t.total = time.perf_counter() - mol_start
    return t


def print_per_molecule_table(results: List[StageTimings]) -> None:
    rows = []
    for r in results:
        rows.append(
            {
                "name": r.name,
                "status": r.status,
                "total_s": round(r.total, 2),
                "embed_s": round(r.embed, 2),
                "mordred_s": round(r.mordred, 2),
                "pharm_s": round(r.pharm, 2),
                "tier": r.embed_tier,
                "embed_fail": r.embed_failures,
                "timeout_fail": r.timeout_failures,
                "n_confs": r.n_confs,
                "n_window": r.n_in_window,
                "mordred/conf_ms": (
                    round(1000 * sum(r.per_conf_mordred) / len(r.per_conf_mordred), 1) if r.per_conf_mordred else 0.0
                ),
            }
        )
    df = pd.DataFrame(rows)
    log.info("\n=== Per-molecule timing (seconds unless noted) ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        log.info(df.to_string(index=False))


def print_aggregate(results: List[StageTimings]) -> None:
    ok = [r for r in results if r.status == "ok"]
    if not ok:
        log.info("\nNo successful runs; skipping aggregate.")
        return

    total = sum(r.total for r in ok)
    embed = sum(r.embed for r in ok)
    energy = sum(r.energies for r in ok)
    mordred = sum(r.mordred for r in ok)
    pharm = sum(r.pharm for r in ok)
    rdkit_shape = sum(r.rdkit_shape for r in ok)

    other = total - (embed + energy + mordred + pharm + rdkit_shape)

    log.info(f"\n=== Aggregate over {len(ok)} successful molecules ===")
    log.info(f"{'Stage':15s} {'total_s':>8s} {'% total':>8s}  {'avg_s/mol':>9s}")
    for label, dt in [
        ("embed+optimize", embed),
        ("energies", energy),
        ("mordred", mordred),
        ("pharmacophore", pharm),
        ("rdkit_shape", rdkit_shape),
        ("other", other),
        ("TOTAL", total),
    ]:
        pct = 100.0 * dt / total if total > 0 else 0.0
        avg = dt / len(ok)
        log.info(f"{label:15s} {dt:8.2f} {pct:7.1f}%  {avg:9.2f}")
    log.info(f"\nThroughput: {len(ok) / total:.2f} mol/s ({total / len(ok):.2f} s/mol)")


def main():
    parser = argparse.ArgumentParser(description="Profile the 3D fast endpoint per-stage")
    parser.add_argument("--n-conformers", type=int, default=10, help="Conformers per molecule (fast default: 10)")
    parser.add_argument("--compounds", type=str, default=None, help="Comma-separated subset of names to run")
    args = parser.parse_args()

    log.info(f"Loading reference compounds from PublicData: {REFERENCE_KEY}")
    df = PublicData().get(REFERENCE_KEY)

    if args.compounds:
        wanted = {s.strip() for s in args.compounds.split(",")}
        df = df[df["name"].isin(wanted)].reset_index(drop=True)
        if df.empty:
            log.error(f"No compounds matched: {wanted}")
            return

    log.info(f"Profiling {len(df)} compounds with n_conformers={args.n_conformers}\n")
    results: List[StageTimings] = []
    for _, row in df.iterrows():
        log.info(f"  [{row['name']:14s}] heavy={row['heavy_atoms']:>3} rot={row['rot_bonds']:>2} ...")
        t = profile_molecule(row["name"], row["smiles"], args.n_conformers)
        log.info(f"    → {t.total:.2f}s (embed={t.embed:.2f}s, mordred={t.mordred:.2f}s)")
        results.append(t)

    print_per_molecule_table(results)
    print_aggregate(results)


if __name__ == "__main__":
    main()
