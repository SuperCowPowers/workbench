"""AWS Glue Job Utilities"""

import sys
from typing import List, Dict, Optional


def get_resolved_options(argv: List[str], options: Optional[List[str]] = None) -> Dict[str, str]:
    """Take the Glue Jobs argv list of args and organize them into a dictionary

    Args:
        argv (list[str]): The Glue Jobs argv list of args
        options (list[str]): The list of options to resolve, defaults to None

    Returns:
        dict: The Glue Jobs argv list of args organized into a dictionary
    """
    resolved_options = {}

    if options is None:
        options = [arg[2:] for arg in argv if arg.startswith("--")]
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            key = argv[i][2:]
            if key in options:
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    resolved_options[key] = argv[i + 1]
                    i += 1  # Increment i for the value part
                else:
                    resolved_options[key] = ""
        i += 1  # Always increment i for the next argument

    return resolved_options


if __name__ == "__main__":
    # Test the glue utils functions
    print("Testing Glue Utils Functions")
    print(get_resolved_options(sys.argv))

    # Test the resolved options
    args = [
        "/tmp/dispatch_test.py",
        "true",
        "--s3path",
        "s3://blah/foo.csv",
        "--job-bookmark-option",
        "job-bookmark-disable",
        "--JOB_ID",
        "j_a123",
        "true",
        "--starting-result-id",
        "200000001",
        "--foo-name",
        "my_name",
        "--JOB_RUN_ID",
        "jr_z456",
        "--JOB_NAME",
        "dispatch_test",
        "--TempDir",
        "s3://aws-glue-assets-123/temporary/",
    ]
    print("All Resolved Options")
    print(get_resolved_options(args))

    # Test the resolved options with a specified list of options
    print("Specified Resolved Options")
    print(get_resolved_options(args, ["s3path", "JOB_ID", "JOB_RUN_ID", "JOB_NAME"]))
