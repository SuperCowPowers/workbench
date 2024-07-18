import pytest

# Try to import the GraphCore class, and if it fails, skip the test
try:
    from sageworks.core.artifacts.graph_core import GraphCore
except ImportError as e:
    pytest.skip(f"Skipping test: {e}", allow_module_level=True)

def test_general_info():
    """Simple test of the Endpoint functionality"""

    test_graph = GraphCore("karate-club")

    # Call the various methods

    # Let's do a check/validation of the Graph
    assert test_graph.exists()

    # Creation/Modification Times
    print(test_graph.created())
    print(test_graph.modified())

    # Details
    print(test_graph.details())

    # Get the tags associated with this Graph
    print(f"Tags: {test_graph.get_tags()}")

if __name__ == "__main__":
    pytest.main()
