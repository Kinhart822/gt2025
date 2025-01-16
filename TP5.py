import numpy as np
import heapq

# Create the adjacency matrix for the given graph G
# Nodes are labeled as A, B, C, D, E, F, G, H, L, M
nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L', 'M']
num_nodes = len(nodes)
node_to_index = {node: index for index, node in enumerate(nodes)}

# Initialize adjacency matrix for Undirected and Weighted graph
adj_matrix = [[float('inf')] * num_nodes for _ in range(num_nodes)]

# Add edges with weights as shown in the graph
edges = [
    ('A', 'C', 1), ('A', 'B', 4),
    ('B', 'F', 3),
    ('C', 'D', 8), ('C', 'F', 7),
    ('D', 'H', 5),
    ('F', 'H', 1), ('F', 'E', 1),
    ('E', 'H', 2),
    ('H', 'G', 3), ('H', 'M', 7), ('H', 'L', 6),
    ('G', 'M', 4),
    ('M', 'L', 1),
    ('L', 'G', 4), ('L', 'E', 2)
]

# Fill the adjacency matrix with the given edges
for edge in edges:
    x, y, weight = edge
    i, j = node_to_index[x], node_to_index[y]
    adj_matrix[i][j] = weight
    adj_matrix[j][i] = weight

# Convert to NumPy array
adj_matrix_np = np.array(adj_matrix)


# Format the output
def format_value(x):
    if x == float('inf'):
        return ' inf'
    return f'{int(x):4d}'


formatted_matrix = np.array2string(
    adj_matrix_np,
    formatter={'all': format_value},
    separator=' ',
    max_line_width=120
)

# Print adjacency matrix
print("Adjacency Matrix for Undirected and Weighted graph:")
print(formatted_matrix)

index_to_node = {index: node for index, node in enumerate(nodes)}

# Dijkstra's Algorithm Implementation
def dijkstra(adj_matrix, source, target):
    source_idx = node_to_index[source]
    target_idx = node_to_index[target]

    # Distance array and priority queue
    distances = [float('inf')] * num_nodes
    distances[source_idx] = 0
    prev = [None] * num_nodes
    pq = [(0, source_idx)]  # (distance, node)

    while pq:
        current_dist, current_node = heapq.heappop(pq)

        # If we've already processed this node with a shorter distance, skip it
        if current_dist > distances[current_node]:
            continue

        # Explore neighbors
        for neighbor in range(num_nodes):
            weight = adj_matrix[current_node][neighbor]
            if weight != float('inf'):  # Check if there's an edge
                new_dist = current_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    prev[neighbor] = current_node
                    heapq.heappush(pq, (new_dist, neighbor))

    # Reconstruct the shortest path
    path = []
    current = target_idx
    while current is not None:
        path.append(index_to_node[current])
        current = prev[current]
    path.reverse()

    return path, distances[target_idx]

# Input source and target nodes
source = input("Enter the source node (A-M): ").strip().upper()
target = input("Enter the target node (A-M): ").strip().upper()

if source in node_to_index and target in node_to_index:
    shortest_path, total_weight = dijkstra(adj_matrix, source, target)
    print(f"Shortest path from {source} to {target}: {' -> '.join(shortest_path)}")
    print(f"Weighted sum of the shortest path: {total_weight}")
else:
    print("Invalid nodes. Please enter valid node labels (A-M).")

