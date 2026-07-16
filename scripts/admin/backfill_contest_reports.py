"""Republish the /contests/* reports with the current contest_report() schema.

Reports published before the ``created``/``contested`` columns were added lack them, and the
Model Contests page (plus its main-page preview) reads those columns directly -- the report is
the contract, so the fix is to republish rather than have the UI paper over the gap. The
promotion arbiter republishes on every run, so this only closes the window between a Workbench
upgrade and each pipeline's next arbiter run.

Rebuilds from the live roster (CachedMeta champion/challenger models) rather than the stale
report contents, so a rebuilt report also picks up any roster change. Idempotent: re-running
just republishes the same reports. Run once per AWS environment.

Usage:
    python scripts/admin/backfill_contest_reports.py            # dry-run (default)
    python scripts/admin/backfill_contest_reports.py --apply    # publish reports
    python scripts/admin/backfill_contest_reports.py --apply --endpoint my-endpoint
"""

import argparse
import sys

from workbench.api import Reports
from workbench.cached.cached_meta import CachedMeta
from workbench.cached.cached_model import CachedModel
from workbench.utils.model_comparison import contest_report


def orphan_reports(reports: Reports, endpoints: set) -> list[str]:
    """Published /contests/* reports whose endpoint no longer has a promotion node. Nothing
    rebuilds these, so they keep their old schema and still break the page."""
    published = {loc.removeprefix("/contests/") for loc in reports.list() if loc.startswith("/contests/")}
    return sorted(published - endpoints)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Publish the reports (default is a dry-run)")
    parser.add_argument("--endpoint", help="Only process this endpoint (default: all champion endpoints)")
    args = parser.parse_args()

    meta = CachedMeta()
    reports = Reports()
    champions = meta.champion_models()
    if champions.empty:
        print("No champion endpoints found.")
        return 0

    all_endpoints = set(champions["Endpoint"])
    if args.endpoint:
        champions = champions[champions["Endpoint"] == args.endpoint]
        if champions.empty:
            print(f"No champion endpoint named '{args.endpoint}'.")
            return 1

    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'} | Contests: {len(champions)}\n")

    published = skipped = failed = 0
    for _, row in champions.iterrows():
        endpoint, champion = row["Endpoint"], row["Model"]
        try:
            challengers = [CachedModel(name) for name in meta.challenger_models(endpoint)]
            report = contest_report(CachedModel(champion), challengers, endpoint)
            if report is None:
                print(f"  - {endpoint}: no metrics for any model, skipped")
                skipped += 1
                continue
            n_challengers = int((report["role"] == "challenger").sum())
            if args.apply:
                reports.upsert(f"/contests/{endpoint}", report)
                print(f"  + {endpoint}: published ({n_challengers} challengers)")
            else:
                print(f"  + {endpoint}: would publish ({n_challengers} challengers)")
            published += 1
        except Exception as e:
            print(f"  ! {endpoint}: FAILED - {e}")
            failed += 1

    verb = "Published" if args.apply else "Would publish"
    print(f"\n{verb}: {published} | Skipped: {skipped} | Failed: {failed}")

    # Orphans keep the old schema and still break the page, so surface them for a manual call
    orphans = orphan_reports(reports, all_endpoints)
    if orphans:
        print(f"\nOrphan reports ({len(orphans)}): no promotion node, so nothing republishes them:")
        for endpoint in orphans:
            print(f"  ? {endpoint}")
        print("Delete them with Reports().delete('/contests/<endpoint>') if the contest is retired.")

    if not args.apply and published:
        print("\nDry-run only. Re-run with --apply to publish the reports.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
