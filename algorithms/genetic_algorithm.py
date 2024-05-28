import networkx as nx
import numpy as np
from typing import Tuple, Callable


class GeneticAlgorithm:
    def __init__(self, graph: nx.graph, population_size: int, generations: int,
                 mutation_probability: float, tournament_size: int, elitism: int,
                 fitness_function: Callable[[nx.graph, np.ndarray], int]):
        # Default values
        self.population = None

        # Assign args
        self.population_size = population_size
        self.generations = generations
        self.mutation_probability = mutation_probability
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.graph = graph
        self.fitness_function = fitness_function

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
            fitness = np.array([
                self.fitness_function(self.graph, individual) for individual in self.population])

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

            self.population = np.array(new_population)

        print(f"The generations have come to an end")

        # After the generations end their life, we must extract the last best values
        fitness = np.array([
            self.fitness_function(self.graph, individual) for individual in self.population])

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

    def __initialize_population__(self):
        self.population = np.random.randint(2, size=(self.population_size, self.graph.number_of_nodes()))

    def __select_individuals__(self) -> Tuple[np.ndarray, np.ndarray]:
        def tournament_selection() -> np.ndarray:
            # We will select the best candidate from a set of random candidates
            participants = np.random.choice(self.population_size, self.tournament_size, replace=False)

            # Get the candidate with the best fitness
            best_index = max(participants,
                             key=lambda x: self.fitness_function(self.graph, self.population[x]))
            return self.population[best_index]

        return tournament_selection(), tournament_selection()

    def __crossover_genes__(self, parent_a: np.ndarray, parent_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # We must avoid the edges of the "genome" to have a good split
        splitting_point = np.random.randint(self.graph.number_of_nodes())

        # We split the parents
        parent_a_split = np.split(parent_a, [splitting_point])
        parent_b_split = np.split(parent_b, [splitting_point])

        # We create the children
        child_a = np.concatenate((parent_a_split[0], parent_b_split[1]), axis=0)
        child_b = np.concatenate((parent_b_split[0], parent_a_split[1]), axis=0)

        return child_a, child_b

    def __mutate__(self, child: np.ndarray) -> np.ndarray:
        # We create a mutation mask using the mutation probability
        mutation = np.random.rand(self.graph.number_of_nodes()) < self.mutation_probability

        # We apply the mask, flipping the bit k where mutation[k] is equal to 1
        child[mutation] = 1 - child[mutation]

        return child
