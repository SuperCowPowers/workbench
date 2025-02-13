from workbench.api.graph_store import GraphStore
from workbench.utils.graph_utils import modified, details, get_tags


def test_general_info():
    """Simple tests of Graph functionality"""
    graph_store = GraphStore()

    # Call the various methods
    assert graph_store.check("test/karate_club")

    test_graph = graph_store.get("test/karate_club")

    # Modification Time
    print(modified(test_graph))

    # Get the tags associated with this Graph
    print(f"Tags: {get_tags(test_graph)}")

    # Details
    print(details(test_graph))


if __name__ == "__main__":

    # Run the tests
    test_general_info()

    print("All tests passed!")
