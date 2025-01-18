import networkx as nx

# Create a graph
G = nx.Graph()

# Add nodes with attributes
G.add_node(1, id=1, name="Alice", age=30)
G.add_node(2, id=2, name="Bob", age=25)

# Print nodes with attributes before serialization
print("Before serialization:")
for node, attrs in G.nodes(data=True):
    print(node, attrs)

# Serialize to JSON
graph_json = nx.readwrite.json_graph.node_link_data(G, edges="edges", name="dummy")
# graph_json = nx.readwrite.json_graph.node_link_data(G, edges="edges")

# Deserialize back
G_deserialized = nx.readwrite.json_graph.node_link_graph(graph_json, edges="edges", name="dummy")
# G_deserialized = nx.readwrite.json_graph.node_link_graph(graph_json, edges="edges")

# Print nodes with attributes after deserialization
print("\nAfter deserialization:")
for node, attrs in G_deserialized.nodes(data=True):
    print(node, attrs)
