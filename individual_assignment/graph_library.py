import heapq
import re
import timeit
import random

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
        start_time = timeit.default_timer()
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
                end_time = timeit.default_timer()
                return path, end_time - start_time
            for neighbors in graph[last]:
                if neighbors[0] not in visited:
                    visited[neighbors[0]] = 1
                    parent[neighbors[0]] = last
                    queue.append(neighbors[0])
        return "path not found"

    def deepth_first_search(self, graph, source, destination):
        start_time = timeit.default_timer()
        stack = []
        visited = {source: 1}
        path = []
        stack.append(source)
        while stack:
            last = stack.pop()
            path.append(last)
            if last == destination:
                end_time = timeit.default_timer()
                return path, end_time - start_time
            else:
                for neighbor in graph[last]:
                    if neighbor[0] not in visited:
                        visited[neighbor[0]] = 1
                        stack.append(neighbor[0])
        return "not found"

    def dijkstras(self, graph, source, destination):
        start_time = timeit.default_timer()
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
                end_time = timeit.default_timer()
                return path, end_time - start_time
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
        start_time = timeit.default_timer()
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
                end_time = timeit.default_timer()
                return paths, end_time - start_time

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


def create_graph(file, heuristic,random_1x,random_2x,random_3x,random_4x):
    graph = Graph()
    graph_1x = Graph()
    def original_graph():
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
        return adj_list,h_nodes

    def generate_1x_graph():

        adj_list_1x = {}
        h_nodes ={}
        orginal_citis = []
        with open(random_1x, "r") as rnd_1x:
            cities = list(map(str, rnd_1x.read().split()))
        with open(random_1x, "r") as hf:
            for city in hf:
                latitude = random.uniform(40,60)
                longtude = random.uniform(40,60)
                node_content = re.split('[\n]', city)
                h_nodes[Node(node_content[0]).name] = (latitude, longtude)
        for key in original_graph()[0]:
            orginal_citis.append(key)
        num_connection = random.randint(1, 8)
        for city1 in cities:
            for i in range(num_connection):
                weight = random.randint(70, 250)
                city2 = random.choice(orginal_citis)
                graph_1x.add_edge(Node(city1), Node(city2), weight)
        for iv, (k, edge) in enumerate(graph_1x.edges.items()):
            if k[0] not in adj_list_1x:
                adj_list_1x[k[0]] = [(k[1], int(edge.weight))]
            else:
                adj_list_1x[k[0]].append((k[1], int(edge.weight)))
        heuristic_nodes = original_graph()[1]
        for h_node in h_nodes:
            heuristic_nodes[h_node] = h_nodes[h_node]
        return adj_list_1x,heuristic_nodes
    def generate_2x_graph():

        adj_list_2x = {}
        h_nodes ={}
        orginal_citis = []
        with open(random_2x, "r") as rnd_1x:
            cities = list(map(str, rnd_1x.read().split()))
        with open(random_2x, "r") as hf:
            for city in hf:
                latitude = random.uniform(50,80)
                longtude = random.uniform(70,80)
                node_content = re.split('[\n]', city)
                h_nodes[Node(node_content[0]).name] = (latitude, longtude)
        for key in original_graph()[0]:
            orginal_citis.append(key)
        num_connection = random.randint(1, 8)
        for city1 in cities:
            for i in range(num_connection):
                weight = random.randint(70, 250)
                city2 = random.choice(orginal_citis)
                graph_1x.add_edge(Node(city1), Node(city2), weight)
        for iv, (k, edge) in enumerate(graph_1x.edges.items()):
            if k[0] not in adj_list_2x:
                adj_list_2x[k[0]] = [(k[1], int(edge.weight))]
            else:
                adj_list_2x[k[0]].append((k[1], int(edge.weight)))
        heuristic_nodes = generate_1x_graph()[1]
        for h_node in h_nodes:
            heuristic_nodes[h_node] = h_nodes[h_node]
        return adj_list_2x,heuristic_nodes
    def generate_3x_graph():

        adj_list_3x = {}
        h_nodes ={}
        orginal_citis = []
        with open(random_3x, "r") as rnd_1x:
            cities = list(map(str, rnd_1x.read().split()))
        with open(random_3x, "r") as hf:
            for city in hf:
                latitude = random.uniform(70,100)
                longtude = random.uniform(70,80)
                node_content = re.split('[\n]', city)
                h_nodes[Node(node_content[0]).name] = (latitude, longtude)
        for key in original_graph()[0]:
            orginal_citis.append(key)
        num_connection = random.randint(1, 8)
        for city1 in cities:
            for i in range(num_connection):
                weight = random.randint(70, 250)
                city2 = random.choice(orginal_citis)
                graph_1x.add_edge(Node(city1), Node(city2), weight)
        for iv, (k, edge) in enumerate(graph_1x.edges.items()):
            if k[0] not in adj_list_3x:
                adj_list_3x[k[0]] = [(k[1], int(edge.weight))]
            else:
                adj_list_3x[k[0]].append((k[1], int(edge.weight)))
        heuristic_nodes = generate_2x_graph()[1]
        for h_node in h_nodes:
            heuristic_nodes[h_node] = h_nodes[h_node]
        return adj_list_3x,heuristic_nodes
    def generate_4x_graph():

        adj_list_4x = {}
        h_nodes ={}
        orginal_citis = []
        with open(random_4x, "r") as rnd_1x:
            cities = list(map(str, rnd_1x.read().split()))
        with open(random_4x, "r") as hf:
            for city in hf:
                latitude = random.uniform(70,100)
                longtude = random.uniform(30,40)
                node_content = re.split('[\n]', city)
                h_nodes[Node(node_content[0]).name] = (latitude, longtude)
        for key in original_graph()[0]:
            orginal_citis.append(key)
        num_connection = random.randint(1, 8)
        for city1 in cities:
            for i in range(num_connection):
                weight = random.randint(70, 250)
                city2 = random.choice(orginal_citis)
                graph_1x.add_edge(Node(city1), Node(city2), weight)
        for iv, (k, edge) in enumerate(graph_1x.edges.items()):
            if k[0] not in adj_list_4x:
                adj_list_4x[k[0]] = [(k[1], int(edge.weight))]
            else:
                adj_list_4x[k[0]].append((k[1], int(edge.weight)))
        heuristic_nodes = generate_3x_graph()[1]
        for h_node in h_nodes:
            heuristic_nodes[h_node] = h_nodes[h_node]
        return adj_list_4x,heuristic_nodes
    def calculate_time(adj_list, function):
        if function == "bfs":
            total_time = 0
            for key in graph.vertices.keys():
                for k in graph.vertices.keys():
                    if k != key:
                        total_time += graph.breadth_first_search(adj_list, k, key)[1]
            return total_time / 380
        if function == "dfs":
            total_time = 0
            for key in graph.vertices.keys():
                for k in graph.vertices.keys():
                    if k != key:
                        total_time += graph.deepth_first_search(adj_list, k, key)[1]
            return total_time / 380
        if function == "dijkstras":
            total_time = 0
            for key in graph.vertices.keys():
                for k in graph.vertices.keys():
                    if k != key:
                        total_time += graph.dijkstras(adj_list, k, key)[1]
            return total_time / 380
        if function == "astar":
            total_time = 0
            for key in graph.vertices.keys():
                for k in graph.vertices.keys():
                    if k != key:
                        total_time += graph.astar_search(adj_list, generate_4x_graph()[1], k, key)[1]
            return total_time / 380

    def calculate_length(adj_list, function):
        if function == "bfs":
            total_length = 0
            for key in graph.vertices.keys():
                for k in graph.vertices.keys():
                    if k != key:
                        total_length += len(graph.breadth_first_search(adj_list, k, key)[0])
            return total_length / 380
        if function == "dfs":
            total_length = 0
            for key in graph.vertices.keys():
                for k in graph.vertices.keys():
                    if k != key:
                        total_length += len(graph.deepth_first_search(adj_list, k, key)[0])
            return total_length / 380
        if function == "dijkstras":
            total_length = 0
            for key in graph.vertices.keys():
                for k in graph.vertices.keys():
                    if k != key:
                        total_length += len(graph.dijkstras(adj_list, k, key)[0])
            return total_length / 380
        if function == "astar":
            total_length = 0
            for key in graph.vertices.keys():
                for k in graph.vertices.keys():
                    if k != key:
                        total_length += len(graph.astar_search(adj_list, generate_4x_graph()[1], k, key)[0])
            return total_length / 380
    average_time = [calculate_time(generate_4x_graph()[0], "bfs"), calculate_time(generate_4x_graph()[0], "dfs"),
                    calculate_time(generate_4x_graph()[0], "dijkstras"), calculate_time(generate_4x_graph()[0], "astar")]
    average_length = [calculate_length(generate_4x_graph()[0], "bfs"), calculate_length(generate_4x_graph()[0], "dfs"),
                      calculate_length(generate_4x_graph()[0], "dijkstras"), calculate_length(generate_4x_graph()[0], "astar")]
    algorithm = ["bfs", "dfs", "dijkstras", "A*"]
    plt.plot(algorithm, average_length)
    plt.xlabel("Search algorithm")
    plt.ylabel("Average Length Taken")
    plt.legend()
    plt.show()


create_graph("edges.text", "heuristic.text","random_1x.text","random_2x.text","random_3x.text","random_4x.text")
