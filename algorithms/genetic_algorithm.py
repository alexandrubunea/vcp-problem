import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from utils.generate_graph import GenerateGraph
from typing import Tuple


class GeneticAlgorithm:
    def __init__(self, graph: GenerateGraph, population_size: int, generations: int,
                 mutation_probability: float, tournament_size: int, elitism: int):
        # Default values
        self.population = None

        # Assign args
        self.population_size = population_size
        self.generations = generations
        self.mutation_probability = mutation_probability
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.graph = graph

        # Init the population
        self.__initialize_population__()

        # Metrics to build graphs
        self.avg_fitness_evolution = []
        self.best_fitness_evolution = []
        self.best_individual_evolution = []

        self.best_fitness_value_oat = None  # aot = off all time
        self.best_individual_oat = None

    def run(self):
        for generation in range(self.generations):
            if not generation % 10:
                print(f"Current generation {generation}/{self.generations}")

            # We calculate the fitness for each individual, in each population
            fitness = np.array([self.__fitness__(individual) for individual in self.population])

            # We extract the best fitness and the best individual so far
            best_fitness = np.max(fitness)
            best_individual = self.population[np.argmax(fitness)]

            # We append those into the metrics
            self.avg_fitness_evolution.append(np.average(fitness))
            self.best_fitness_evolution.append(best_fitness)
            self.best_individual_evolution.append(best_individual)

            new_population = []

            # Keep the elites in the future generation
            sorted_indices_mask = np.argsort(fitness)[::-1]  # create a mask with indices sorted by fitness
            self.population = self.population[sorted_indices_mask]  # apply the mask
            new_population.extend(self.population[:self.elitism])  # move the elites into the new generation

            # We reproduce the current population
            while len(new_population) < self.population_size:
                parent_a, parent_b = self.__select_individuals__()  # We get two parents at random
                child_a, child_b = self.__crossover_genes__(parent_a, parent_b)  # Crossover the genes

                # The children may mutate
                child_a = self.__mutate__(child_a)
                child_b = self.__mutate__(child_b)

                # We add the children to the new population
                new_population.append(child_a)

                if len(new_population) < self.population_size:
                    new_population.append(child_b)

            assert len(new_population) == self.population_size, \
                f"Expected population size {self.population_size}, but got {len(new_population)}"

            self.population = np.array(new_population)

        print(f"The generations have come to an end")

        # After the generations end their life, we must extract the last best values
        fitness = np.array([self.__fitness__(individual) for individual in self.population])

        best_fitness = np.max(fitness)
        best_individual = self.population[np.argmax(fitness)]

        self.avg_fitness_evolution.append(np.average(fitness))
        self.best_fitness_evolution.append(best_fitness)
        self.best_individual_evolution.append(best_individual)

        # And in the final, from the best fitness values during the evolution, we extract the biggest one
        self.best_fitness_value_oat = max(self.best_fitness_evolution)
        self.best_individual_oat = max(zip(
            self.best_fitness_evolution,
            self.best_individual_evolution), key=lambda x: x[0])[1]

    def show_metrics(self):
        # We will create plots to show the metrics

        # Avg fitness per generation
        plt.plot(self.avg_fitness_evolution)
        plt.xlabel("Generation")
        plt.ylabel("Avg fitness")
        plt.title("Avg fitness in every generation")
        plt.show()

        # Best fitness per generation
        plt.plot(self.best_fitness_evolution)
        plt.xlabel("Generation")
        plt.ylabel("Best fitness")
        plt.title("Best fitness in every generation")
        plt.show()

        # We show a table that contains the best individual in every generation
        genetic_evolution_table = PrettyTable(["Generation", "Individual"])
        for generation, individual in enumerate(self.best_individual_evolution):
            genetic_evolution_table.add_row([generation, individual])
        print(genetic_evolution_table)

    def show_solution(self):
        # Extract the nodes and the edges from the solution
        vertices = np.where(self.best_individual_oat == 1)[0]  # Extract the vertices
        edges = self.__extract_edges_from_vertices(vertices)

        vertices_color = ["red" if vertex in vertices else "skyblue" for vertex in self.graph.get_graph().nodes()]
        edges_color = ["green" if edge in edges else "black" for edge in self.graph.get_graph().edges()]
        pos = nx.spring_layout(self.graph.get_graph())

        nx.draw(self.graph.get_graph(), pos, node_color=vertices_color, edge_color=edges_color, with_labels=True)
        plt.show()

        # Print the corresponding values
        print(f"The best individual is {self.best_individual_oat} with a fitness of {self.best_fitness_value_oat}")

    def __initialize_population__(self):
        self.population = np.random.randint(2, size=(self.population_size, self.graph.get_graph().number_of_nodes()))

    def __extract_edges_from_vertices(self, vertices) -> set[tuple[int, int]]:
        edges = set()

        for vertex in vertices:
            for neighbour in self.graph.get_graph().neighbors(vertex):
                if (vertex, neighbour) in self.graph.get_graph().edges() or \
                        (neighbour, vertex) in self.graph.get_graph().edges():
                    # We add the edge that we covered, keeping the order in our set to avoid duplicates.
                    edges.add((vertex, neighbour) if vertex < neighbour else (neighbour, vertex))

        return edges

    def __fitness__(self, individual: np.ndarray) -> int:
        selected_vertices = np.where(individual == 1)[0]  # We extract the active vertex from the current individual
        covered_edges = self.__extract_edges_from_vertices(selected_vertices)

        return len(covered_edges) - len(selected_vertices)  # We must penalize according to the number of used vertices

    def __select_individuals__(self) -> Tuple[np.ndarray, np.ndarray]:
        def tournament_selection() -> np.ndarray:
            # We will select the best candidate from a set of random candidates
            participants = np.random.choice(self.population_size, self.tournament_size, replace=False)

            # Get the candidate with the best fitness
            best_index = max(participants, key=lambda x: self.__fitness__(self.population[x]))
            return self.population[best_index]

        return tournament_selection(), tournament_selection()

    def __crossover_genes__(self, parent_a: np.ndarray, parent_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # We must avoid the edges of the "genome" to have a good split
        splitting_point = np.random.randint(self.graph.get_graph().number_of_nodes())

        # We split the parents
        parent_a_split = np.split(parent_a, [splitting_point])
        parent_b_split = np.split(parent_b, [splitting_point])

        # We create the children
        child_a = np.concatenate((parent_a_split[0], parent_b_split[1]), axis=0)
        child_b = np.concatenate((parent_b_split[0], parent_a_split[1]), axis=0)

        return child_a, child_b

    def __mutate__(self, child: np.ndarray) -> np.ndarray:
        # We create a mutation mask using the mutation probability
        mutation = np.random.rand(self.graph.get_graph().number_of_nodes()) < self.mutation_probability

        # We apply the mask, flipping the bit k where mutation[k] is equal to 1
        child[mutation] = 1 - child[mutation]

        return child
