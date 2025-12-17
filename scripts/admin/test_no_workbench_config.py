import os

os.environ.pop("WORKBENCH_CONFIG", None)

try:
    from workbench.api import Meta

    print(Meta)
except Exception as e:
    # Print out the exception
    print("error: ", e)
    print("Workbench API is not available")
finally:
    # Print out the exception
    print("Finally: Workbench API is not available")

print("Do Stuff")
