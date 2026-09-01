"""Check whether the feature (molecular descriptor) endpoints are stale.

Any endpoint whose last modification is more than two weeks old should be
rebuilt so it picks up the current Workbench image.

Usage:
    python scripts/admin/feature_endpoint_update_check.py
"""

from datetime import datetime, timedelta, timezone

from workbench.api import Endpoint

FEATURE_ENDPOINTS = [
    "smiles-to-2d-v1",
    "smiles-to-2d-salt-v1",
    "smiles-to-3d-v1",
    "smiles-to-3d-v2",
    "smiles-to-2d-3d-v1",
    "smiles-to-2d-3d-v2",
    "smiles-to-2d-3d-salt-v2",
    "smiles-to-fingerprints-v1",
]

STALE_AFTER = timedelta(weeks=2)


def main():
    now = datetime.now(timezone.utc)
    stale = []

    for name in FEATURE_ENDPOINTS:
        end = Endpoint(name)
        if not end.exists():
            print(f"{name:28} NOT FOUND")
            continue

        modified = end.modified()
        age = now - modified
        days = age.total_seconds() / 86400
        status = "NEEDS UPDATE" if age > STALE_AFTER else "OK"
        if age > STALE_AFTER:
            stale.append(name)
        print(f"{name:28} {modified:%Y-%m-%d %H:%M} UTC  ({days:5.1f} days)  {status}")

    print("=" * 70)
    if stale:
        print(f"{len(stale)} endpoint(s) need to be updated:")
        for name in stale:
            print(f"  - {name}")
    else:
        print("All feature endpoints are up to date.")


if __name__ == "__main__":
    main()
