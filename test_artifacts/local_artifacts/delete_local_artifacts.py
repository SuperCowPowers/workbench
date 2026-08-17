"""This Script Deletes the Workbench Local Artifacts used for the tests

Deleting a LocalModel takes its endpoints with it, so the endpoints are not
listed here. Deleting artifacts by hand (rm -rf) skips that cascade and leaves
endpoints pointing at a model directory that no longer exists.
"""

from workbench.local import LocalDataSource, LocalFeatureSet, LocalModel

if __name__ == "__main__":

    # Delete the local test Artifacts
    LocalModel("local-test-regression").delete()
    LocalFeatureSet("local_test_features").delete()
    LocalDataSource("local_test_data").delete()
