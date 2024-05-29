from typing import List
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def show_metrics(avg_fitness_evolution: List[int], best_fitness_evolution: List[int],
                 best_individual_evolution: List[List[int]]) -> None:
    # We will create plots to show the metrics

    # Avg fitness per generation
    plt.plot(avg_fitness_evolution)
    plt.xlabel("Generation")
    plt.ylabel("Avg fitness")
    plt.title("Avg fitness in every generation")
    plt.show()

    # Best fitness per generation
    plt.plot(best_fitness_evolution)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("Best fitness in every generation")
    plt.show()

    # We show a table that contains the best individual in every generation
    genetic_evolution_table = PrettyTable(["Generation", "Individual"])
    for generation, individual in enumerate(best_individual_evolution):
        genetic_evolution_table.add_row([generation, individual])
    print(genetic_evolution_table)


def show_solution(graph: nx.graph, best_individual_oat: np.ndarray, best_fitness_value_oat: int) -> None:
    # Extract the nodes from the solution
    vertices = np.where(best_individual_oat == 1)[0]  # Extract the vertices
    edges = set()

    for vertex in vertices:
        for neighbour in graph.neighbors(vertex):
            if (vertex, neighbour) in graph.edges():
                # We add the edge that we covered, keeping the order in our set to avoid duplicates.
                edges.add((vertex, neighbour))
                edges.add((neighbour, vertex))

    vertices_color = ["red" if vertex in vertices else "skyblue" for vertex in graph.nodes()]
    edges_color = ["green" if edge in edges else "black" for edge in graph.edges()]
    pos = nx.spring_layout(graph)

    nx.draw(graph, pos, edge_color=edges_color, node_color=vertices_color, with_labels=True)
    plt.show()

    # Print the corresponding values
    print(f"The best individual is {best_individual_oat} with a fitness of {best_fitness_value_oat}")


def compare_metrics(metrics_a: list, metrics_b: list, name_a: str, name_b: str, title: str) -> None:
    plt.plot(metrics_a, label=name_a)
    plt.plot(metrics_b, label=name_b)

    plt.title(title)
    plt.legend()
    plt.show()
