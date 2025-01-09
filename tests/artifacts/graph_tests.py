from workbench.utils.graph_utils import load_graph, exists, modified, details, get_tags


def test_general_info():
    """Simple test of the Endpoint functionality"""

    # Call the various methods
    assert exists("karate_club")

    test_graph = load_graph("karate_club")

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
