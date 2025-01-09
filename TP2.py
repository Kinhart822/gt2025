def count_components(directed_g):
    def dfs(node_index, compo, graph, visited):
        stack = [node_index]
        while stack:
            current_node = stack.pop()
            if current_node not in visited:
                visited.add(current_node)
                compo.add(current_node + 1)
                to_extend = []
                for j in range(len(graph[current_node])):
                    if graph[current_node][j] == 1 and j not in visited:
                        to_extend.append(j)
                stack.extend(to_extend)

    undirected_g = []
    for i in range(len(directed_g)):
        row = []
        for j in range(len(directed_g)):
            if directed_g[i][j] or directed_g[j][i]:
                row.append(1)
            else:
                row.append(0)
        undirected_g.append(row)

    # Strongly connected components in directed graph
    directed_visited = set()
    directed_components = []
    for i in range(len(directed_g)):
        if i not in directed_visited:
            component = set()
            dfs(i, component, directed_g, directed_visited)
            directed_components.append(component)

    # Weakly connected components in undirected graph
    undirected_visited = set()
    undirected_components = []
    for i in range(len(undirected_g)):
        if i not in undirected_visited:
            component = set()
            dfs(i, component, undirected_g, undirected_visited)
            undirected_components.append(component)

    return "Strongly Connected Components (Strong): " + str(directed_components) + "\nWeakly Connected Components (Weak): " + str(undirected_components)


if __name__ == '__main__':
    G = [
        [0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    result = count_components(G)
    print(result)
