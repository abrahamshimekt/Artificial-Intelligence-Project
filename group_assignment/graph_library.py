import heapq
import re
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from math import radians, cos, sin, asin, sqrt


class Node:

    def __init__(self, name):
        self.name = name
        self.edge_list = []

    def connect(self, node):
        con = (self.name, node.name)
        self.edge_list.append(con)


class Edge:

    def __init__(self, left, right, weight):
        self.left = left
        self.right = right
        self.weight = weight


class Graph:

    def __init__(self):
        self.vertices = {}
        self.edges = {}

    def add_node(self, node):
        self.vertices[node.name] = node

    def add_edge(self, left, right, weight):
        if left.name not in self.vertices:
            self.vertices[left.name] = left

        if right.name not in self.vertices:
            self.vertices[right.name] = right

        edge = Edge(left, right, weight)

        key = (left.name, right.name)
        self.edges[key] = edge

        key = (right.name, left.name)
        self.edges[key] = edge

        left.connect(right)
        right.connect(left)

    def dijkstras(self, graph, function, source, destination):
        nodes_distance = defaultdict(lambda: float('inf'))
        visited = {}
        nodes_distance[source] = 0
        heap = []
        heapq.heappush(heap, (0, source))
        parent = {source: None}
        while len(heap) != 0:
            (shortest_distance, current_node) = heapq.heappop(heap)
            if current_node == destination:
                if function == "closenness":
                    return shortest_distance
                elif function == "betweenness":
                    path = []
                    while current_node:
                        path.append(current_node)
                        current_node = parent.get(current_node, None)
                    return path
            for neighbor in graph[current_node]:
                if neighbor[0] not in visited:
                    prev_distance = nodes_distance[neighbor[0]]
                    curr_distance = nodes_distance[current_node] + neighbor[1]
                    if curr_distance < prev_distance:
                        heapq.heappush(heap, (curr_distance, neighbor[0]))
                        nodes_distance[neighbor[0]] = curr_distance
                        parent[neighbor[0]] = current_node
        return "not found"

    def calculate_distance(self, source_lat, source_lon, destination_lat, destination_lon):
        source_lon, source_lat, destination_lon, destination_lat = map(radians,
                                                                       [source_lon, source_lat, destination_lon,
                                                                        destination_lat])
        distance_lon = destination_lon - source_lon
        distance_lat = destination_lat - source_lat
        a = sin(distance_lat / 2) ** 2 + cos(source_lat) * cos(destination_lat) * sin(distance_lon / 2) ** 2
        c = 2 * asin(sqrt(a))
        km = 6371 * c
        return km

    def astar_search(self, graph, h_nodes, function, source, destination):
        h_fn = {}
        for key in h_nodes:
            h_fn[key] = self.calculate_distance(h_nodes[key][0], h_nodes[key][1], h_nodes[destination][0],
                                                h_nodes[destination][1])

        visited_uninspected = {source: 1}
        visited_inspected = {}
        distance = {source: 0}
        parents = {source: source}
        while len(visited_uninspected) > 0:
            node = None
            for node_visited in visited_uninspected:
                if node is None or distance[node_visited] + h_fn[node_visited] < distance[node] + h_fn[node]:
                    node = node_visited
            if node is None:
                return 'Path does not exist!'
            if node == destination:
                if function == "closenness":
                    return distance[node]
                elif function == "betweenness":
                    paths = []
                    while parents[node] != node:
                        paths.append(node)
                        node = parents[node]
                    paths.append(source)
                    return paths
            for (node_connected, weight) in graph[node]:
                if node_connected not in visited_uninspected and node_connected not in visited_inspected:
                    visited_uninspected[node_connected] = 1
                    parents[node_connected] = node
                    distance[node_connected] = distance[node] + weight
                else:
                    if distance[node_connected] > distance[node] + weight:
                        distance[node_connected] = distance[node] + weight
                        parents[node_connected] = node

                        if node_connected in visited_inspected:
                            visited_inspected.pop(node_connected)
                            visited_uninspected[node_connected] = 1
            visited_uninspected.pop(node)
            visited_inspected[node] = 1
        return 'Path does not exist!'


def create_graph(file, heuristic):
    graph = Graph()
    adj_list = {}
    heuristic_nodes = {}

    with open(file, "r") as ef:
        for edges in ef:
            edges_content = re.split('[:\n]', edges)
            graph.add_edge(Node(edges_content[0]), Node(edges_content[1]), edges_content[2])
    with open(heuristic, "r") as hf:
        for nodes in hf:
            node_content = re.split('[:\n]', nodes)
            heuristic_nodes[Node(node_content[0]).name] = (float(node_content[1]), float(node_content[2]))
    for iv, (k, edge) in enumerate(graph.edges.items()):
        if k[0] not in adj_list:
            adj_list[k[0]] = [(k[1], int(edge.weight))]
        else:
            adj_list[k[0]].append((k[1], int(edge.weight)))
    """This function calculate degree of centrality using a formula:
     No. adjacent edge/No. node -1"""

    def calculate_degree(adj_list, solution_node):
        degree_nodes = {}
        for node in adj_list:
            degree_nodes[node] = len(adj_list[node]) / (len(adj_list) - 1)
        return degree_nodes[solution_node]

    """This function calculates the closeness centrality using a formula:
     No. node -1/the total shortest distances to the solution node """

    def calculate_closenness_dijkstras(closenness, adj_list, solution_node):
        total_shortest_distance = 0
        for node in adj_list:
            total_shortest_distance += graph.dijkstras(adj_list, closenness, node, solution_node)
        return (len(adj_list) - 1) / total_shortest_distance

    """This function calculates the betweenness centrality by using a formula:
    No.shortest paths that pass through the solution node/Total number shortest pass from any node"""

    def calculate_betweenness_dijkstras(betweenness, adj_list, solution_node):
        shortest_paths = []
        for node in adj_list:
            for n in adj_list:
                if n != node:
                    shortest_paths.append(graph.dijkstras(adj_list, betweenness, n, node))
        paths_through = 0
        for element in shortest_paths:
            for i in range(len(element)):
                if element[i] == solution_node and i != 0 and i != len(element) - 1:
                    paths_through += 1
        return paths_through / len(shortest_paths)

    def calculate_closenness_astar(adj_list, heuristic_nodes, closenness, solution_node):
        total_shortest_distance = 0
        for node in adj_list:
            total_shortest_distance += graph.astar_search(adj_list, heuristic_nodes, closenness, node, solution_node)
        return (len(adj_list) - 1) / total_shortest_distance

    def calculate_betweenness_astar(adj_list, heuristic_nodes, betweenness, solution_node):
        shortest_paths = []
        for node in adj_list:
            for n in adj_list:
                if n != node:
                    shortest_paths.append(graph.astar_search(adj_list, heuristic_nodes, betweenness, n, node))
        paths_through = 0
        for element in shortest_paths:
            for i in range(len(element)):
                if element[i] == solution_node and i != 0 and i != len(element) - 1:
                    paths_through += 1
        return paths_through / len(shortest_paths)

    print("degree:", calculate_degree(adj_list, "Bucharest"))
    print("closenness_dijkstra:", calculate_closenness_dijkstras("closenness", adj_list, "Bucharest"))
    print("betweenness_dijkstra:", calculate_betweenness_dijkstras("betweenness", adj_list, "Bucharest"))
    print("closenness_astar:", calculate_closenness_astar(adj_list, heuristic_nodes, "closenness", "Bucharest"))
    print("betweenness_astar:", calculate_betweenness_astar(adj_list, heuristic_nodes, "betweenness", "Bucharest"))


create_graph("edges.text", "heuristic.text")
