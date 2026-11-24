"""Check that every endpoint under a MetaEndpoint is up, and report how each is deployed.

A child passes when its status is InService. Its transport (real-time, serverless, async)
and instance type are printed for information.

Usage:
    python scripts/admin/meta_endpoint_children_check.py [meta_endpoint_name]
"""

import sys

from workbench.api import Endpoint
from workbench.api.meta_endpoint import MetaEndpoint


def check_endpoint(name: str) -> tuple[str, str]:
    """Return (summary, problem) for one child; problem is empty when it is up."""
    end = Endpoint(name)
    if not end.exists():
        return "", "does not exist"
    status = end.aws_meta()["EndpointStatus"]
    transport = "serverless" if end.is_serverless() else "async" if end.is_async() else "real-time"
    problem = "" if status == "InService" else f"status is {status}"
    return f"{status:12} {transport:11} {end.instance_type}", problem


def main():
    meta_name = sys.argv[1] if len(sys.argv) > 1 else "all-regression-1"
    children = sorted(set(MetaEndpoint(meta_name).get_dag().endpoints.values()))

    failed = []
    for name in children:
        summary, problem = check_endpoint(name)
        if problem:
            failed.append(name)
        print(f"{name:40} {summary:45} {'FAIL: ' + problem if problem else 'OK'}")

    print("=" * 70)
    if failed:
        print(f"{meta_name}: {len(failed)} of {len(children)} children are not up:")
        for name in failed:
            print(f"  {name}")
        sys.exit(1)
    print(f"{meta_name}: all {len(children)} children are up.")


if __name__ == "__main__":
    main()
