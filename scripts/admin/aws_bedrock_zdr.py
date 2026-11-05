"""Check Bedrock zero data retention (ZDR) for the Claude models, and set it where missing.

Bedrock retention is a per-region account setting, not a per-model one. A model is covered
by ZDR when the mode is `none` in every region its `us.` inference profile routes to. A region
that was never configured reports `inherit`, which is not ZDR.

Needs admin credentials (bedrock:ListInferenceProfiles, bedrock:GetAccountDataRetention,
bedrock:PutAccountDataRetention). Plain boto3 on the caller's AWS_PROFILE: importing
Workbench would load WORKBENCH_CONFIG and swap in its non-admin profile.

Dry run by default; --apply sets `none` in each region that lacks it.

Usage:
    AWS_PROFILE=scp_sandbox_admin python scripts/admin/aws_bedrock_zdr.py
    AWS_PROFILE=scp_sandbox_admin python scripts/admin/aws_bedrock_zdr.py --apply
    AWS_PROFILE=scp_sandbox_admin python scripts/admin/aws_bedrock_zdr.py --models claude-opus-5-5
"""

import argparse
import sys

import boto3
from botocore.exceptions import ClientError

PROFILE_PREFIX = "us.anthropic."
ZDR_MODE = "none"


def claude_profiles(client) -> dict[str, list[str]]:
    """Every US Claude inference profile, mapped to the regions it routes to."""
    profiles = {}
    for page in client.get_paginator("list_inference_profiles").paginate():
        for p in page["inferenceProfileSummaries"]:
            if p["inferenceProfileId"].startswith(PROFILE_PREFIX):
                # Model ARNs look like arn:aws:bedrock:<region>::foundation-model/<id>
                regions = {m["modelArn"].split(":")[3] for m in p["models"]}
                profiles[p["inferenceProfileId"].removeprefix(PROFILE_PREFIX)] = sorted(regions)
    return profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", help="Bare model names, e.g. claude-opus-5-5 (default: all Claude)")
    parser.add_argument("--region", help="Region to list inference profiles from (default: the profile's region)")
    parser.add_argument("--apply", action="store_true", help="Set retention to 'none' where it isn't")
    args = parser.parse_args()

    session = boto3.Session()
    identity = session.client("sts").get_caller_identity()
    print(f"Account {identity['Account']}  ({identity['Arn']})")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN (pass --apply to set)'}\n")

    profiles = claude_profiles(session.client("bedrock", region_name=args.region or session.region_name))
    missing = [m for m in args.models or [] if m not in profiles]
    if missing:
        sys.exit(f"No {PROFILE_PREFIX} inference profile for: {', '.join(missing)}")
    models = args.models or sorted(profiles)

    # Region -> models routed there, so each region is checked once
    region_models: dict[str, list[str]] = {}
    for model in models:
        for region in profiles[model]:
            region_models.setdefault(region, []).append(model)

    failed = False
    for region, routed in sorted(region_models.items()):
        client = session.client("bedrock", region_name=region)
        mode = client.get_account_data_retention()["mode"]
        if mode == ZDR_MODE:
            print(f"  OK       {region}: {mode}  ({len(routed)} models)")
            continue
        if not args.apply:
            print(f"  NOT ZDR  {region}: {mode}  ({', '.join(routed)})")
            failed = True
            continue
        try:
            new_mode = client.put_account_data_retention(mode=ZDR_MODE)["mode"]
            print(f"  SET      {region}: {mode} -> {new_mode}")
        except ClientError as e:
            print(f"  FAILED   {region}: {mode}  ({e.response['Error']['Message']})")
            failed = True

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
