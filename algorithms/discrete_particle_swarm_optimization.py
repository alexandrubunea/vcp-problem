import numpy as np
import networkx as nx


class Particle:
    def __init__(self, position: int, velocity: int):
        self.position = position
        self.velocity = velocity


class DSPO:
    def __init__(self, graph: nx.graph, population_size: int):
        # Default values
        self.population = None

        # Assign args
        self.graph = graph
        self.population_size = population_size

        # Init the population
        self.__initialize_population__()

    def __initialize_population__(self):
        self.population = np.random.randint(2, size=(self.population_size, self.graph.get_graph().number_of_nodes()))
