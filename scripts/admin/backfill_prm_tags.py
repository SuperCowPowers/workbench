"""Backfill the AWS Partner Revenue Measurement (PRM) attribution tag on existing SageMaker resources.

Workbench tags endpoints, feature groups, and training jobs at creation. This applies the tag once to
endpoints and feature groups that predate that (training jobs are ephemeral). Resources carrying the
legacy ``aws-apn-id`` key are left as they are; AWS still attributes them.

Idempotent. Run once per AWS environment.

Usage:
    python scripts/admin/backfill_prm_tags.py            # dry-run (default)
    python scripts/admin/backfill_prm_tags.py --apply    # write tags
"""

import argparse
import sys

from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.aws_utils import PRM_LEGACY_TAG_KEY, PRM_TAG_KEY, PRM_TAG_VALUE


def resource_arns(sm) -> dict[str, list[str]]:
    """Endpoint and feature group ARNs in the account/region."""
    endpoints = sm.get_paginator("list_endpoints").paginate()
    groups = sm.get_paginator("list_feature_groups").paginate()
    return {
        "endpoint": [e["EndpointArn"] for page in endpoints for e in page["Endpoints"]],
        "feature group": [g["FeatureGroupArn"] for page in groups for g in page["FeatureGroupSummaries"]],
    }


def current_tags(sm, arn: str) -> dict:
    """All AWS tags on the ARN (paginated so we don't miss the tag past the first page)."""
    tags, token = {}, None
    while True:
        kwargs = {"ResourceArn": arn}
        if token:
            kwargs["NextToken"] = token
        resp = sm.list_tags(**kwargs)
        tags.update({t["Key"]: t["Value"] for t in resp.get("Tags", [])})
        token = resp.get("NextToken")
        if not token:
            return tags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Write tags (default is a dry-run)")
    args = parser.parse_args()

    sm = AWSAccountClamp().boto3_session.client("sagemaker")
    print(f"PRM tag: {PRM_TAG_KEY}={PRM_TAG_VALUE}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}\n")

    tagged = already = failed = 0
    for kind, arns in resource_arns(sm).items():
        for arn in arns:
            name = f"{kind} {arn.split('/')[-1]}"
            try:
                tags = current_tags(sm, arn)
                if PRM_TAG_VALUE in (tags.get(PRM_TAG_KEY), tags.get(PRM_LEGACY_TAG_KEY)):
                    already += 1
                    continue
                if args.apply:
                    sm.add_tags(ResourceArn=arn, Tags=[{"Key": PRM_TAG_KEY, "Value": PRM_TAG_VALUE}])
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
