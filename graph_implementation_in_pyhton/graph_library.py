import re
from queue import Queue
from heapq import heapify, heappush, heappop
import sys


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
        visited = set()
        queue_of_nodes = Queue()
        path = []
        path_found = False
        visited.add(source)
        queue_of_nodes.put(source)
        while queue_of_nodes:
            node = queue_of_nodes.get()
            path.append(node)
            if node == destination:
                path_found = True
                break
            else:
                for neighbor in graph[node]:
                    if neighbor[0] not in visited:
                        visited.add(neighbor[0])
                        queue_of_nodes.put(neighbor[0])

        return path if path_found else "not found"

    def deepth_first_search(self, graph, source, destination):
        visited = set()
        stack = []
        path = []
        path_found = False
        visited.add(source)
        stack.append(source)
        while stack:
            node = stack.pop()
            path.append(node)
            if node == destination:
                path_found = True
                break
            else:
                for neighbor in graph[node]:
                    if neighbor[0] not in visited:
                        visited.add(neighbor[0])
                        stack.append(neighbor[0])
        return path if path_found else "not found"

    def dijkstras(self, nodes, graph, source, destination):
        nodes[source]["distance"] = 0
        visited = []
        temp = source
        for i in range(len(graph)):
            if temp not in visited:
                visited.append(source)
                queue_of_nodes = []
                for neighbor in graph[temp]:
                    if neighbor not in visited:
                        distance = nodes[source]["distance"] + neighbor[1]
                        if distance < nodes[neighbor[0]]["distance"]:
                            nodes[neighbor[0]]["distance"] = distance
                            nodes[neighbor[0]]["path"] = nodes[temp]["path"] + list(temp)
                        heappush(queue_of_nodes,(nodes[neighbor[0]]["distance"],neighbor[0]))
                heapify(queue_of_nodes)
                temp =queue_of_nodes[0][1]
        return nodes[destination]["path"] + [destination]

    def heuristic_fun(self,h_nodes,node):
        return h_nodes[node]


    def astrix_search(self,graph,h_nodes,source,destination):
        visited_uninspected= {source}
        visited_inspected = set([])
        distance = {source:0}
        parents = {source:source}
        while len(visited_uninspected) > 0:
            node = None
            for node_visited in visited_uninspected:
                if node is None or distance[node_visited] + self.heuristic_fun(h_nodes, node_visited) < distance[node] + self.heuristic_fun(h_nodes, node):
                    node = node_visited
            if node is None:
                return 'Path does not exist!'
            if node == destination:
                paths= []

                while parents[node] != node:
                    paths.append(node)
                    node= parents[node]

                paths.append(source)

                paths.reverse()
                return paths
            for (node_connected, weight) in graph[node]:
                if node_connected not in visited_uninspected and node_connected not in visited_inspected:
                    visited_uninspected.add(node_connected)
                    parents[node_connected] = node
                    distance[node_connected] = distance[node] + weight
                else:
                    if distance[node_connected] > distance[node] + weight:
                        distance[node_connected] = distance[node] + weight
                        parents[node_connected] = node

                        if node_connected in visited_inspected:
                            visited_inspected.remove(node_connected)
                            visited_uninspected.add(node_connected)
            visited_uninspected.remove(node)
            visited_inspected.add(node)
        return 'Path does not exist!'


def create_graph(file):
    inf = sys.maxsize
    graph = Graph()
    adj_list = {}
    nodes = {}
    h_nodes = {}

    with open(file, "r") as ef:
        for edges in ef:
            edges_content = re.split('[:\n]', edges)
            graph.add_edge(Node(edges_content[0]), Node(edges_content[1]), edges_content[2])

    for iv, (k, edge) in enumerate(graph.edges.items()):
        if k[0] not in adj_list:
            adj_list[k[0]] = [(k[1], int(edge.weight))]
        else:
            adj_list[k[0]].append((k[1],int(edge.weight)))
    for key in graph.vertices.keys():
        h_nodes[key] = 1
        nodes[key] = {"distance": inf, "path": []}
    print(graph.breadth_first_search(adj_list,"Mehadia","Rimniscu"))
    print(graph.deepth_first_search(adj_list, "Mehadia", "Rimniscu"))
    print(graph.dijkstras(nodes,adj_list,"Mehadia","Rimniscu"))
    print(graph.astrix_search(adj_list,h_nodes,"Mehadia","Rimniscu"))
create_graph("edges.text")
