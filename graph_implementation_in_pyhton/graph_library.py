import re
from graph import Node
from graph import Graph


def create_graph(edge_file):
    adj_list = {}
    g = Graph()
    with open(edge_file, "r") as ef:
        for edges in ef:
            edges_content = re.split('[:\n]', edges)
            g.add_edge(Node(edges_content[0]), Node(edges_content[1]), edges_content[2])
    for edge in g.edges.keys():
        if edge[0] not in adj_list:
            adj_list[edge[0]] = [edge[1]]
        else:
            adj_list[edge[0]].append(edge[1])
    print(adj_list)
create_graph("edges")
