"""List in-progress SageMaker training jobs and what each one is actually doing.

Jobs sitting in the ``Pending`` secondary status are waiting on AWS capacity for
their instance type — they burn wall-clock without training. Those sort first.

Usage:
    python scripts/admin/training_job_status.py
    python scripts/admin/training_job_status.py --waiting     # only capacity-blocked jobs
    AWS_PROFILE=<profile> python scripts/admin/training_job_status.py
"""

import argparse

from workbench.utils.batch_utils import running_training_jobs


def main(waiting_only: bool):
    df = running_training_jobs()
    if waiting_only and not df.empty:
        df = df[df["waiting"]]
    if df.empty:
        print("No matching in-progress training jobs.")
        return

    print(df.drop(columns=["status"]).to_string(index=False))

    blocked = df[df["waiting"]]["name"].tolist()
    if blocked:
        print(f"\n{len(blocked)} job(s) waiting on capacity. Stop one with:")
        print(f"    aws sagemaker stop-training-job --training-job-name {blocked[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--waiting", action="store_true", help="only show jobs waiting on capacity")
    args = parser.parse_args()
    main(args.waiting)
