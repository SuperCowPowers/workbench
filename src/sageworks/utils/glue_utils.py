"""AWS Glue Job Utilities"""

import sys
from typing import List, Dict, Optional
import awswrangler as wr
import logging
import traceback
from contextlib import contextmanager


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


@contextmanager
def exception_log_forward():
    """Context manager to log exceptions to the sageworks logger"""
    log = logging.getLogger("sageworks")
    try:
        yield
    except Exception as e:
        # Capture the stack trace as a list of frames
        tb = e.__traceback__
        # Convert the stack trace into a list of formatted strings
        stack_trace = traceback.format_exception(e.__class__, e, tb)
        # Find the frame where the context manager was entered
        cm_frame = traceback.extract_tb(tb)[0]
        # Filter out the context manager frame
        filtered_stack_trace = []
        for frame in traceback.extract_tb(tb):
            if frame != cm_frame:
                filtered_stack_trace.append(frame)
        # Format the filtered stack trace
        formatted_stack_trace = "".join(traceback.format_list(filtered_stack_trace))
        log.critical("Exception:\n%s%s", formatted_stack_trace, "".join(stack_trace[-1:]))
        raise


def list_s3_files(s3_path: str, extensions: str = "*.csv") -> List[str]:
    """
    Lists files in an S3 path with specified extension.

    Args:
    s3_path (str): The full S3 path (e.g., 's3://my-bucket/my-prefix/').
    extensions (str): File extension to filter by, defaults to '*.csv'.

    Returns:
    List[str]: A list of file paths matching the extension in the S3 path.
    """
    files = wr.s3.list_objects(path=s3_path, suffix=extensions.lstrip("*"))
    return files


if __name__ == "__main__":
    # Test the glue utils functions
    print("Testing Glue Utils Functions")
    print(get_resolved_options(sys.argv))
    print(list_s3_files("s3://sageworks-public-data/common"))

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

    # Test the exception handler
    with exception_log_forward():
        raise ValueError("Testing the exception handler")
