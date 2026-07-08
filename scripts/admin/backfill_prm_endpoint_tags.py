"""Backfill the AWS Partner Revenue Measurement (PRM) attribution tag on existing endpoints.

New/redeployed endpoints get ``aws-apn-id``=``pc:<product-code>`` at creation time (see
workbench.core.transforms.model_to_endpoint). Endpoints that predate that change need the
tag applied once so AWS attributes their consumption to the ADMET Workbench Marketplace
listing. Training jobs are ephemeral and self-tag going forward, so only endpoints need a
backfill.

Idempotent: re-running skips endpoints that already carry the correct tag. Run once per AWS
environment (product code is the same across all envs).

Usage:
    python scripts/admin/backfill_prm_endpoint_tags.py            # dry-run (default)
    python scripts/admin/backfill_prm_endpoint_tags.py --apply    # write tags
    python scripts/admin/backfill_prm_endpoint_tags.py --apply --endpoint my-endpoint
"""

import argparse
import sys

from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.cached.cached_meta import CachedMeta
from workbench.utils.aws_utils import AWS_MARKETPLACE_PRODUCT_CODE

TAG_KEY = "aws-apn-id"
TAG_VALUE = f"pc:{AWS_MARKETPLACE_PRODUCT_CODE}"


def endpoint_names(sm_client, only: str | None) -> list[str]:
    """Endpoint names to process: a single --endpoint, or all known to Workbench."""
    if only:
        return [only]
    df = CachedMeta().endpoints()
    return [] if df.empty else df["Name"].tolist()


def current_tags(sm_client, arn: str) -> dict:
    """All AWS tags on the ARN (paginated so we don't miss the tag past the first page)."""
    tags, token = {}, None
    while True:
        kwargs = {"ResourceArn": arn}
        if token:
            kwargs["NextToken"] = token
        resp = sm_client.list_tags(**kwargs)
        tags.update({t["Key"]: t["Value"] for t in resp.get("Tags", [])})
        token = resp.get("NextToken")
        if not token:
            return tags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Write tags (default is a dry-run)")
    parser.add_argument("--endpoint", help="Only process this endpoint (default: all endpoints)")
    args = parser.parse_args()

    sm = AWSAccountClamp().boto3_session.client("sagemaker")
    names = endpoint_names(sm, args.endpoint)
    if not names:
        print("No endpoints found.")
        return 0

    print(f"PRM tag: {TAG_KEY}={TAG_VALUE}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'} | Endpoints: {len(names)}\n")

    tagged = already = failed = 0
    for name in names:
        try:
            arn = sm.describe_endpoint(EndpointName=name)["EndpointArn"]
            if current_tags(sm, arn).get(TAG_KEY) == TAG_VALUE:
                print(f"  = {name}: already tagged")
                already += 1
                continue
            if args.apply:
                sm.add_tags(ResourceArn=arn, Tags=[{"Key": TAG_KEY, "Value": TAG_VALUE}])
                print(f"  + {name}: tagged")
            else:
                print(f"  + {name}: would tag")
            tagged += 1
        except Exception as e:
            print(f"  ! {name}: FAILED - {e}")
            failed += 1

    verb = "Tagged" if args.apply else "Would tag"
    print(f"\n{verb}: {tagged} | Already tagged: {already} | Failed: {failed}")
    if not args.apply and tagged:
        print("\nDry-run only. Re-run with --apply to write the tags.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
