"""Verify the shared pipeline_utils package is importable in this run.

Creates no AWS artifacts, so it's a cheap end-to-end check of the staging path:
local runs import it off the launcher's PYTHONPATH, Batch runs import it from the
copy the runner stages out of S3.
"""

from pipeline_utils.smoke import add, describe_environment


def main():
    print("pipeline_utils imported successfully")
    for key, value in describe_environment().items():
        print(f"  {key}: {value}")

    result = add(2, 3)
    assert result == 5, f"expected 5, got {result}"
    print(f"  add(2, 3): {result}")


if __name__ == "__main__":
    main()
