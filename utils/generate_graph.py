import networkx as nx

# Honestly, we can just do it fine without this class, but now I am too lazy to refactor it


class GenerateGraph:
    def __init__(self, number_of_vertices: int, probability_of_edge: float):
        self.graph = nx.gnp_random_graph(number_of_vertices, probability_of_edge)

    def get_graph(self) -> nx.Graph:
        return self.graph
