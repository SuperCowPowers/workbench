import os

os.environ.pop("SAGEWORKS_CONFIG", None)

try:
    from sageworks.meta import Meta
except Exception as e:
    # Print out the exeception
    print("error: ", e)
    print("SageWorks API is not available")
finally:
    # Print out the exeception
    print("Finally: SageWorks API is not available")

print("Do Stuff")
