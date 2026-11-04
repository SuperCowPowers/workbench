"""Pick a Butina cutoff for PXR by matching the CV gap to the deployment gap.

A split exists so out-of-fold error estimates deployment error, and the quantity that
governs both is how far a prediction row sits from the rows that trained it. This measures
that distance -- each row's max Tanimoto to its training set -- for the real deployment
(phase-1 test vs train) and for the CV folds every split strategy produces on the train
rows, then ranks the strategies by how closely their fold distribution matches deployment.

Fingerprints are the ones Butina clusters on (`similarity_fingerprints`: Morgan r=2 counts,
MinMax Tanimoto), so cutoffs read in the same units as the distances here. `butina_cutoff`
is a Tanimoto DISTANCE: 0.6 clusters at similarity >= 0.4, and a higher cutoff makes larger
clusters and a harder split.

Columns:
    groups / singletons / top_group  group count, singleton fraction, largest group's share
    fold_min / fold_max              smallest and largest fold (groups stay intact)
    p25 / p50 / p75                  quantiles of each validation row's max sim to its train fold
    w1_test                          Wasserstein distance to the test-vs-train distribution

Usage:
    python pxr_split_cutoff.py
    python pxr_split_cutoff.py --cutoffs 0.5 0.6 0.7 --plot split_cutoff.png
"""

import argparse

import numpy as np
from rdkit import DataStructs
from scipy.stats import wasserstein_distance
from workbench.api import FeatureSet
from workbench.training.splits import get_butina_clusters, get_scaffold_groups, get_split_indices
from workbench.utils.chem_utils.fingerprints import similarity_fingerprints

FS_NAME = "openadmet_pxr_f1"
N_FOLDS = 5
SEED = 42


def similarity_matrix(query_fps: list, ref_fps: list) -> np.ndarray:
    """Dense Tanimoto matrix, rows = query, columns = reference."""
    return np.array([DataStructs.BulkTanimotoSimilarity(fp, ref_fps) for fp in query_fps], dtype=np.float32)


def fold_nn_similarity(sim: np.ndarray, folds: list) -> np.ndarray:
    """Each row's max similarity to the training rows of the fold that holds it out."""
    nn = np.empty(sim.shape[0], dtype=np.float32)
    for train_idx, val_idx in folds:
        nn[val_idx] = sim[np.ix_(val_idx, train_idx)].max(axis=1)
    return nn


def group_stats(groups: np.ndarray) -> dict:
    """Group count, singleton fraction and largest group's share of rows."""
    _, counts = np.unique(groups, return_counts=True)
    return {
        "groups": len(counts),
        "singletons": float((counts == 1).sum() / len(groups)),
        "top_group": float(counts.max() / len(groups)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cutoffs", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8])
    parser.add_argument("--plot", help="Write overlaid CDFs of the max-similarity distributions to this PNG")
    args = parser.parse_args()

    df = FeatureSet(FS_NAME).pull_dataframe()
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "phase1_test"].reset_index(drop=True)
    print(f"{len(train)} train rows, {len(test)} phase-1 test rows")

    train_fps, train_pos = similarity_fingerprints(train["smiles"].tolist())
    test_fps, _ = similarity_fingerprints(test["smiles"].tolist())
    if len(train_pos) != len(train):
        raise ValueError(f"{len(train) - len(train_pos)} train SMILES failed to parse; positions would misalign")

    test_nn = similarity_matrix(test_fps, train_fps).max(axis=1)
    train_sim = similarity_matrix(train_fps, train_fps)

    smiles = train["smiles"].tolist()
    strategies = [("random", None), ("scaffold", get_scaffold_groups(smiles))]
    strategies += [(f"butina_{c:g}", get_butina_clusters(smiles, cutoff=c)) for c in args.cutoffs]

    rows, dists = [], {"phase1_test": test_nn}
    for name, groups in strategies:
        strategy, _, cutoff = name.partition("_")
        folds = get_split_indices(
            train,
            n_splits=N_FOLDS,
            strategy=strategy,
            random_state=SEED,
            butina_cutoff=float(cutoff) if cutoff else 0.6,
        )
        nn = fold_nn_similarity(train_sim, folds)
        dists[name] = nn
        sizes = [len(v) for _, v in folds]
        stats = (
            group_stats(groups) if groups is not None else {"groups": len(train), "singletons": 1.0, "top_group": 0.0}
        )
        p25, p50, p75 = np.quantile(nn, [0.25, 0.5, 0.75])
        rows.append(
            {
                "strategy": name,
                **stats,
                "fold_min": min(sizes),
                "fold_max": max(sizes),
                "p25": p25,
                "p50": p50,
                "p75": p75,
                "w1_test": wasserstein_distance(nn, test_nn),
            }
        )

    t25, t50, t75 = np.quantile(test_nn, [0.25, 0.5, 0.75])
    print(f"\nphase-1 test -> train max sim: p25 {t25:.3f}  p50 {t50:.3f}  p75 {t75:.3f}\n")
    header = f"{'strategy':<13}{'groups':>7}{'singl':>7}{'top':>7}{'f_min':>7}{'f_max':>7}"
    header += f"{'p25':>7}{'p50':>7}{'p75':>7}{'w1_test':>9}"
    print(header)
    for r in rows:
        line = f"{r['strategy']:<13}{r['groups']:>7}{r['singletons']:>7.2f}{r['top_group']:>7.2f}"
        line += f"{r['fold_min']:>7}{r['fold_max']:>7}{r['p25']:>7.3f}{r['p50']:>7.3f}{r['p75']:>7.3f}"
        line += f"{r['w1_test']:>9.3f}"
        print(line)
    best = min(rows, key=lambda r: r["w1_test"])
    print(f"\nClosest to deployment: {best['strategy']} (w1 {best['w1_test']:.3f})")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, nn in dists.items():
            xs = np.sort(nn)
            style = {"color": "black", "linewidth": 3} if name == "phase1_test" else {"linewidth": 1.5}
            ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), label=name, **style)
        ax.set_xlabel("Max Tanimoto to training rows (Morgan r=2 counts)", fontsize=12)
        ax.set_ylabel("Cumulative fraction of rows", fontsize=12)
        ax.set_title("PXR: CV fold gap vs deployment gap", fontsize=13)
        ax.legend(fontsize=11)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
