class Node:

    def __init__(self, name):
        self.name = name
        self.edge_list = []

    def connect(self, node):
        con = (self.name, node.name)
        self.edge_list.append(con)


class Edge:

    def __init__(self, left, right, weight=1):
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

        e = Edge(left, right, weight)

        key = (left.name, right.name)
        self.edges[key] = e

        key = (right.name, left.name)
        self.edges[key] = e

        left.connect(right)
        right.connect(left)

    def search(self, a, b):
        pass

    def to_aj_matrix(self):
        pass


