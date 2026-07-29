"""Audit the /contests/* reports against the live artifacts they reference.

Contest reports are published only by the promotion arbiter (model_promotion.py) and
nothing ever deletes them, so a renamed or retired endpoint leaves an orphaned report
behind that still renders on the Model Contests page. This finds those.

Uses live Meta() rather than CachedMeta(): the meta cache has a 30s TTL, and the whole
point here is to compare the reports against ground truth.

Run this when no promotion Batch jobs are in flight. The arbiter tears an endpoint down
to redeploy it and deletes the dethroned model, so a contest caught mid-promotion will
be reported as an orphan.

Usage:
    python scripts/admin/contest_audit.py                  # audit everything
    python scripts/admin/contest_audit.py --orphans-only    # only contests with issues
    python scripts/admin/contest_audit.py --prune           # delete orphans (confirms first)
"""

import argparse
import sys

from workbench.api import Meta, Reports

CONTEST_PREFIX = "/contests/"


def live_artifacts(meta: Meta) -> tuple[dict, set]:
    """Ground truth from live metadata: {endpoint_name: serving_model} and the model names."""
    endpoints = meta.endpoints(details=True)
    models = meta.models(details=True)
    endpoint_map = dict(zip(endpoints["Name"], endpoints["Input"]))
    model_names = set(models["Model Group"])
    return endpoint_map, model_names


def audit_contest(df, endpoint_map: dict, model_names: set) -> dict:
    """Audit one contest report. Returns a row dict with an "issues" list."""
    endpoint = df["endpoint"].iloc[0]
    champion_rows = df[df["role"] == "champion"]
    champion = champion_rows["model"].iloc[0] if not champion_rows.empty else None

    issues = []
    # The orphan signal: no endpoint means nothing will ever republish this report
    if endpoint not in endpoint_map:
        issues.append("endpoint missing")
    if champion is None:
        issues.append("no champion row")
    elif champion not in model_names:
        # Not fatal on its own -- the arbiter deliberately retires dethroned models
        issues.append("champion model missing")
    if endpoint in endpoint_map and champion and endpoint_map[endpoint] != champion:
        issues.append(f"endpoint serves {endpoint_map[endpoint]}")

    return {
        "endpoint": endpoint,
        "champion": champion,
        "challengers": int((df["role"] == "challenger").sum()),
        "contested": bool(df.iloc[0].get("contested", False)),
        "scored": df["timestamp"].max(),
        "issues": issues,
    }


def audit_all(reports: Reports, endpoint_map: dict, model_names: set) -> dict:
    """Audit every published contest. Returns {contest_name: row dict}."""
    results = {}
    for location in sorted(loc for loc in reports.list() if loc.startswith(CONTEST_PREFIX)):
        name = location.removeprefix(CONTEST_PREFIX)
        df = reports.get(location)
        if df is None or df.empty:
            results[name] = {
                "endpoint": None,
                "champion": None,
                "challengers": 0,
                "contested": False,
                "scored": None,
                "issues": ["empty report"],
            }
            continue
        results[name] = audit_contest(df, endpoint_map, model_names)
    return results


def print_results(results: dict, orphans_only: bool):
    """Print the audit table."""
    rows = {k: v for k, v in results.items() if v["issues"]} if orphans_only else results
    if not rows:
        print("No contests to report.")
        return
    width = max(len(k) for k in rows)
    for name, row in rows.items():
        status = "OK" if not row["issues"] else "; ".join(row["issues"])
        marker = "  " if not row["issues"] else " !"
        print(f"{marker} {name:<{width}}  champion={row['champion']}  {status}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--orphans-only", action="store_true", help="Only show contests with issues")
    parser.add_argument("--prune", action="store_true", help="Delete reports whose endpoint no longer exists")
    args = parser.parse_args()

    reports = Reports()
    endpoint_map, model_names = live_artifacts(Meta())
    results = audit_all(reports, endpoint_map, model_names)

    total = len(results)
    flagged = sum(1 for r in results.values() if r["issues"])
    print(f"Contests: {total} | Clean: {total - flagged} | Flagged: {flagged}\n")
    print_results(results, args.orphans_only)

    # Only a missing endpoint justifies deletion -- a missing champion model is normal
    orphans = [name for name, row in results.items() if "endpoint missing" in row["issues"]]
    if not orphans:
        return 0

    print(f"\nOrphans ({len(orphans)}): endpoint gone, so nothing will ever republish these:")
    for name in orphans:
        print(f"  ? {name}")

    if not args.prune:
        print("\nRe-run with --prune to delete them.")
        return 0

    answer = input(f"\nDelete {len(orphans)} contest report(s)? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        print("Aborted.")
        return 0

    for name in orphans:
        reports.delete(f"{CONTEST_PREFIX}{name}")
        print(f"  x {name}: deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
