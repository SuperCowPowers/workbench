"""Redeploy serverless endpoints as real-time endpoints with data capture, keeping name and tags.

Each endpoint is redeployed from its own model with to_endpoint(serverless=False, data_capture=True),
which auto-selects the instance type. A redeploy removes the endpoint's S3 artifacts other than
data_capture/ (inference captures, cross-fold results), so re-run any held-out captures after.

Usage:
    python scripts/admin/serverless_to_realtime.py            # dry-run (default)
    python scripts/admin/serverless_to_realtime.py --apply    # redeploy
"""

import argparse

from workbench.api import Endpoint, Model

SERVERLESS_ENDPOINTS = [
    "mdr1-pappab-reg-1",
    "mdr1-pappab-class-1",
    "ppb-rat-free-reg-1",
    "ppb-rat-free-class-1",
    "logp-value-reg-1",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Redeploy (default is a dry-run)")
    args = parser.parse_args()

    for name in SERVERLESS_ENDPOINTS:
        end = Endpoint(name)
        if not end.exists():
            print(f"{name}: not found, skipping")
            continue
        if not end.is_serverless():
            print(f"{name}: already real-time ({end.instance_type}), skipping")
            continue

        model_name = end.get_input()
        tags = end.get_tags()
        print(f"{name}: model={model_name} tags={tags}")
        if not args.apply:
            continue

        end = Model(model_name).to_endpoint(name=name, tags=tags, serverless=False, data_capture=True)
        print(f"{name}: now real-time on {end.instance_type}, data_capture={end.data_capture().is_enabled()}")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to redeploy.")


if __name__ == "__main__":
    main()
