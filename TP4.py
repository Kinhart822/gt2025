import numpy as np

# Number of nodes in the graph
num_nodes = 9

# Initialize adjacency matrix for Undirected and Weighted graph
adj_matrix = np.full((num_nodes, num_nodes), float('inf'))

# Add edges (from node X to Y with weight W)
edges = [
    (1, 2, 4), (1, 5, 1), (1, 7, 2),
    (2, 3, 7), (2, 6, 5),
    (3, 4, 1), (3, 6, 8),
    (4, 6, 6), (4, 7, 4), (4, 8, 3),
    (5, 6, 9), (5, 7, 10),
    (6, 9, 2),
    (7, 9, 8),
    (8, 9, 1),
    (9, 8, 7)
]

# Fill the adjacency matrix with the given edges
for edge in edges:
    x, y, weight = edge
    adj_matrix[x - 1][y - 1] = weight


# Format the output
def format_value(x):
    if x == float('inf'):
        return ' inf'
    return f'{int(x):4d}'


formatted_matrix = np.array2string(
    adj_matrix,
    formatter={'all': format_value},
    separator=' ',
    max_line_width=120
)

# Print adjacency matrix
print("Adjacency Matrix for Undirected and Weighted graph:")
print(formatted_matrix)


# Function to implement Prim's Algorithm
def prim_algorithm(adj_matrix, num_nodes, root):
    selected_nodes = [False] * num_nodes
    selected_nodes[root] = True
    edges = []
    total_weight = 0

    while len(edges) < num_nodes - 1:
        min_edge = (None, None, float('inf'))  # (x, y, weight)
        for u in range(num_nodes):
            if selected_nodes[u]:
                for v in range(num_nodes):
                    if not selected_nodes[v] and adj_matrix[u][v] != float('inf'):
                        if adj_matrix[u][v] < min_edge[2]:
                            min_edge = (u, v, adj_matrix[u][v])

        u, v, weight = min_edge
        edges.append((u + 1, v + 1, weight))
        total_weight += weight
        selected_nodes[v] = True

    return edges, total_weight


# Kruskal's Algorithm requires disjoint set (Union-Find) data structure
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1


# Function to implement Kruskal's Algorithm
def kruskal_algorithm(adj_matrix, num_nodes):
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i][j] != float('inf'):
                edges.append((i, j, adj_matrix[i][j]))

    edges.sort(key=lambda x: x[2])  # Sort edges by weight
    disjoint_set = DisjointSet(num_nodes)
    mst_edges = []
    total_weight = 0

    for u, v, weight in edges:
        if disjoint_set.find(u) != disjoint_set.find(v):
            disjoint_set.union(u, v)
            mst_edges.append((u + 1, v + 1, weight))
            total_weight += weight

    return mst_edges, total_weight


# Input from the user
root_node = int(input("\nEnter the root node for Prim's algorithm: ")) - 1

# Run Prim's algorithm & Kruskal's algorithm
prim_edges, prim_weight = prim_algorithm(adj_matrix, num_nodes, root_node)
kruskal_edges, kruskal_weight = kruskal_algorithm(adj_matrix, num_nodes)

# Print the results for Prim's algorithm and Kruskal's algorithm
print("\nPrim's Algorithm MST:")
for edge in prim_edges:
    print(f"Edge: {edge[0]} - {edge[1]}, Weight: {edge[2]}")
print(f"Total weight of MST: {prim_weight}")

print("\nKruskal's Algorithm MST:")
for edge in kruskal_edges:
    print(f"Edge: {edge[0]} - {edge[1]}, Weight: {edge[2]}")
print(f"Total weight of MST: {kruskal_weight}")
