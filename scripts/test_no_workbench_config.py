import os

os.environ.pop("WORKBENCH_CONFIG", None)

try:
    from workbench.api import Meta
except Exception as e:
    # Print out the exeception
    print("error: ", e)
    print("Workbench API is not available")
finally:
    # Print out the exeception
    print("Finally: Workbench API is not available")

print("Do Stuff")
