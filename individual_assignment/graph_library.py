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

    def breadth_first_search(self, graph, source, destination):
        queue = deque()
        visited = {source: 1}
        parent = {}
        queue.append(source)
        while queue:
            last = queue.popleft()
            if last == destination:
                node = last
                path = []
                while node:
                    path.append(node)
                    node = parent.get(node, None)
                return len(path)
            for neighbors in graph[last]:
                if neighbors[0] not in visited:
                    visited[neighbors[0]] = 1
                    parent[neighbors[0]] = last
                    queue.append(neighbors[0])
        return "path not found"

    def deepth_first_search(self, graph, source, destination):
        stack = []
        visited = {source: 1}
        path = []
        stack.append(source)
        while stack:
            last = stack.pop()
            path.append(last)
            if last == destination:
                return len(path)
            else:
                for neighbor in graph[last]:
                    if neighbor[0] not in visited:
                        visited[neighbor[0]] = 1
                        stack.append(neighbor[0])
        return "not found"

    def dijkstras(self, graph, source, destination):
        nodes_distance = defaultdict(lambda: float('inf'))
        visited = {}
        nodes_distance[source] = 0
        heap = []
        heapq.heappush(heap, (0, source))
        parent = {source: None}
        while len(heap) != 0:
            (shortest_distance, current_node) = heapq.heappop(heap)
            if current_node == destination:
                node = current_node
                path = []
                while node:
                    path.append(node)
                    node = parent.get(node, None)
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

    def astar_search(self, graph, h_nodes, source, destination):
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
    global second_dj, first_dj, second_dfs, first_dfs, second_bfs, first_bfs, second_as, first_as
    graph = Graph()
    adj_list = {}
    h_nodes = {}

    with open(file, "r") as ef:
        for edges in ef:
            edges_content = re.split('[:\n]', edges)
            graph.add_edge(Node(edges_content[0]), Node(edges_content[1]), edges_content[2])
    with open(heuristic, "r") as hf:
        for nodes in hf:
            node_content = re.split('[:\n]', nodes)
            h_nodes[Node(node_content[0]).name] = (float(node_content[1]), float(node_content[2]))
    for iv, (k, edge) in enumerate(graph.edges.items()):
        if k[0] not in adj_list:
            adj_list[k[0]] = [(k[1], int(edge.weight))]
        else:
            adj_list[k[0]].append((k[1], int(edge.weight)))

    dist_taken = []
    bfs = 0
    for key in graph.vertices.keys():
        for k in graph.vertices.keys():
            if k != key:
                bfs += graph.breadth_first_search(adj_list, k, key)
    dist_taken.append(bfs / 380)
    dfs = 0
    for key in graph.vertices.keys():
        for k in graph.vertices.keys():
            if k != key:
                dfs += graph.deepth_first_search(adj_list, k, key)
    dist_taken.append(dfs / 380)

    dj = 0
    for key in graph.vertices.keys():
        for k in graph.vertices.keys():
            if k != key:
                dj += graph.dijkstras(adj_list, k, key)

    dist_taken.append(dj / 380)

    astar = 0
    for key in graph.vertices.keys():
        for k in graph.vertices.keys():
            if k != key:
                astar += graph.astar_search(adj_list, h_nodes, k, key)
    dist_taken.append(astar / 380)
    search_algorithms = ["bfs", "dfs", "dijkstras", "astar"]
    plt.bar(search_algorithms, dist_taken)
    plt.suptitle('Graph search algorithms')
    plt.xlabel("search algorithms")
    plt.ylabel("Average time taken")
    plt.show()


create_graph("edges.text", "heuristic.text")
