import networkx as nx


def in_context_path(G):
    nodes_in_path = []
    while len(G) > 0:
        min_degree_node = min(G.nodes(), key=G.degree)
        nodes_in_path.append(min_degree_node)
        inorder = list(sorted(list(G[min_degree_node].items()), key=lambda x: x[1]['weight']))
        G.remove_node(min_degree_node)
        for node, _ in inorder:
            nodes_in_path.append(node)
            G.remove_node(node)
    return nodes_in_path


if __name__ == '__main__':
    # Create an empty undirected graph
    G = nx.Graph()

    # Add edges with weights
    G.add_edge('A', 'B', weight=4)
    G.add_edge('B', 'C', weight=2)
    G.add_edge('A', 'C', weight=3)
    G.add_edge('B', 'D', weight=3)
    G.add_edge('E', 'F', weight=3)
    G.add_edge('F', 'D', weight=3)

    # G.add_weighted_edges_from([('A', 'B', 4), ('B', 'C', 2), ('A', 'C', 3)])
    print(in_context_path(G))
