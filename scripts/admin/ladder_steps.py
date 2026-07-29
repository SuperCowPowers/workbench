"""Find Batch jobs whose training dropped down the instance ladder.

A Batch job that submitted more than one training job for the same model hit a capacity
timeout on a rung and moved to the next one (see INSTANCE_LADDERS in features_to_model).
The earlier attempts show as Stopped on the instance that never came free.

Usage:
    python scripts/admin/ladder_steps.py                  # all Batch jobs in the window
    python scripts/admin/ladder_steps.py chemprop_hpo     # name substring filter
    AWS_PROFILE=<profile> python scripts/admin/ladder_steps.py chemprop_hpo
"""

import argparse

from workbench.utils.batch_utils import batch_jobs, batch_job_training_jobs, training_job_status


def main(name_filter: str):
    names = batch_jobs(name_filter)["name"].tolist()
    print(f"Checking {len(names)} Batch job(s) for ladder steps...")

    stepped = 0
    for name in names:
        training = batch_job_training_jobs(name)
        if len(training) < 2:
            continue
        stepped += 1
        print(f"\n{name}  <-- ladder stepped {len(training)}x")
        for job in training:
            status = training_job_status(job)
            if status is None:
                print(f"    {job}  (no longer in SageMaker)")
                continue
            print(f"    {job}  {status['status']:<10} {status['instance']:<16} {status['age']}")

    print(f"\n{stepped} of {len(names)} Batch job(s) stepped down the ladder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name_filter", nargs="?", help="case-insensitive substring filter on the Batch job name")
    args = parser.parse_args()
    main(args.name_filter)
