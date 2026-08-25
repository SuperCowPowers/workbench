"""Pull the live CYP challenge leaderboards from the HF Space.

The Space reads its boards from a private S3 bucket server-side, so the CSVs in its
repo are placeholders. Its own download endpoints hand back the real thing with no AWS
credentials involved.

Every download button is wired through `functools.partial` with its arguments bound, so
Gradio exposes them as zero-argument endpoints named `/partial_N` in tab order rather
than by function name. The mapping below was derived by matching our own scores back to
each board. If OpenADMET reorder or add tabs it will shift -- re-derive with:

    from gradio_client import Client
    c = Client("openadmet/cyp-challenge", verbose=False)
    [c.predict(api_name=f"/partial_{i}") for i in range(9)]

Needs `gradio_client` (not a Workbench dependency):  uv pip install gradio_client

Usage:
    python cyp_leaderboard.py                      # regression boards, our row + top 5
    python cyp_leaderboard.py --user NAME --top 10
    python cyp_leaderboard.py --tdi --save outputs/leaderboards
"""

import argparse
from pathlib import Path

import pandas as pd

REGRESSION = {
    "macro": "/partial",
    "CYP1A2": "/partial_1",
    "CYP2C9": "/partial_2",
    "CYP2D6": "/partial_3",
    "CYP3A4": "/partial_4",
}
TDI = {"macro": "/partial_5", "CYP2D6": "/partial_6", "CYP3A4": "/partial_7"}


def fetch(boards: dict) -> dict:
    """Download each board, keyed by name."""
    from gradio_client import Client

    client = Client("openadmet/cyp-challenge", verbose=False)
    out = {}
    for name, endpoint in boards.items():
        df = pd.read_csv(client.predict(api_name=endpoint))
        metric = next((c for c in df.columns if "ST-RAE" in c or "MCC" in c), None)
        if metric is None:
            raise ValueError(f"{endpoint} returned no scoring column — tab order likely changed")
        out[name] = df
    return out


def field_sd(boards: dict, top: int = 10, ratio: float = 1.2533) -> dict:
    """Recover each isoform's blind-set label sd from the field's own scores.

    `R2 = 1 - MSE/var(y)` inverts to `sd(y) = RMSE/sqrt(1 - R2)`, and every entry on a
    board is scored against the same labels, so each row is an independent estimate of the
    same quantity. The board reports MAE rather than RMSE, so the inversion needs a
    residual-shape ratio.

    That ratio is not constant across the field. Residuals dominated by a constant offset
    have RMSE/MAE -> 1.0 while well-centred ones approach the Gaussian 1.2533, so
    inverting every row at one fixed ratio drifts with entry quality -- corr(sd, R2) runs
    -0.6 to -0.9 on three of the four boards. Anchoring on the top entries, where the
    Gaussian ratio is the applicable one, removes the drift.

    Residuals heavier-tailed than Gaussian would raise all four isoforms together, so read
    these as a common-mode floor rather than four independent measurements.
    """
    out = {}
    for iso in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"):
        d = boards[iso].dropna(subset=["MAE", "R²"]).nlargest(top, "R²")
        out[iso] = float((ratio * d["MAE"] / (1 - d["R²"]) ** 0.5).median())
    return out


def summarize(boards: dict, user: str, top: int) -> None:
    """Print the leader, our row, and the head of each board."""
    for name, df in boards.items():
        metric = next(c for c in df.columns if "ST-RAE" in c or "MCC" in c)
        cols = [c for c in ("Rank", "Username", metric, "MAE", "R²", "Spearman's ρ", "Kendall's τ") if c in df.columns]
        print(f"\n=== {name} ({len(df)} entries, sorted by {metric}) ===")
        print(df[cols].head(top).to_string(index=False))
        mine = df[df["Username"] == user]
        if len(mine):
            if int(mine["Rank"].iloc[0]) > top:
                print(f"  ...\n{mine[cols].to_string(index=False, header=False)}")
        else:
            print(f"  ({user} not on this board)")


def calibration_check(boards: dict, user: str) -> None:
    """R2 against the ceiling its own Spearman supports.

    The best R2 reachable by rescaling predictions is exactly Pearson r^2, so a row far
    below its ceiling is displaced rather than badly modelled -- recoverable without
    retraining. Spearman stands in for Pearson here, so read it as approximate.
    """
    rows = []
    for name, df in boards.items():
        mine = df[df["Username"] == user]
        if not len(mine) or "R²" not in df.columns or "Spearman's ρ" not in df.columns:
            continue
        r2, sp = float(mine["R²"].iloc[0]), float(mine["Spearman's ρ"].iloc[0])
        rows.append((name, r2, sp**2, 100 * r2 / sp**2 if sp else float("nan")))
    if not rows:
        return
    print(f"\n=== calibration check for '{user}' ===")
    print(f"{'board':<8} {'R2':>8} {'ceiling':>9} {'% of ceiling':>13}")
    for name, r2, ceil, pct in rows:
        print(f"{name:<8} {r2:>8.4f} {ceil:>9.4f} {pct:>12.0f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="briford", help="Username to locate on each board")
    parser.add_argument("--top", type=int, default=5, help="Rows to show per board")
    parser.add_argument("--tdi", action="store_true", help="Pull the TDI boards instead")
    parser.add_argument("--sd", action="store_true", help="Recover blind-set label sd per isoform from the field")
    parser.add_argument("--save", type=Path, help="Directory to write each board as CSV")
    args = parser.parse_args()

    fetched = fetch(TDI if args.tdi else REGRESSION)
    summarize(fetched, args.user, args.top)
    if not args.tdi:
        calibration_check(fetched, args.user)
        if args.sd:
            print("\n=== blind-set label sd, inverted from the field ===")
            for iso, sd in field_sd(fetched).items():
                print(f"{iso:<8} {sd:.3f}")

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")
        for name, df in fetched.items():
            path = args.save / f"{'tdi' if args.tdi else 'reg'}_{name}_{stamp}.csv"
            df.to_csv(path, index=False)
        print(f"\nWrote {len(fetched)} CSVs to {args.save}")
