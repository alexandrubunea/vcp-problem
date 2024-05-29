import random
import time
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.discrete_particle_swarm_optimization import DPSO
import networkx as nx
import numpy as np
from utils.metrics import compare_metrics, show_solution


def perforamnce_test(tests: int, vertices_lower_bound: int, vertices_higher_bound: int, population_size: int,
                     iterations: int):

    results_ga = []
    results_dpso = []

    runtime_ga = []
    runtime_dpso = []

    for i in range(tests):
        graph = nx.gnp_random_graph(random.randint(vertices_lower_bound, vertices_higher_bound), 0.3)
        print(f"==================[TEST {i}]==================")
        print(f"NUMBER OF VERTICES: {graph.number_of_nodes()}\n\n")

        def fitness_func(solution: np.ndarray) -> int:
            selected_vertices = np.where(solution == 1)[0]
            covered_edges = set()
            for vertex in selected_vertices:
                for neighbour in graph.neighbors(vertex):
                    if (vertex, neighbour) in graph.edges() or \
                            (neighbour, vertex) in graph.edges():
                        covered_edges.add((vertex, neighbour) if vertex < neighbour else (neighbour, vertex))
            fitnesz = len(covered_edges) - len(selected_vertices)
            components = list(nx.connected_components(graph))
            if len(components) > 1:
                for component in components:
                    exists_at_least_one = False
                    for vertex in component:
                        if vertex in selected_vertices:
                            exists_at_least_one = True

                    if not exists_at_least_one:
                        fitnesz -= 1337 * 100
                        break
            if len(covered_edges) != graph.number_of_edges():
                fitnesz -= 1337 * 100
            return fitnesz

        ga = GeneticAlgorithm(graph.number_of_nodes(), population_size, iterations,
                              0.3, 5, 2, fitness_func)
        dpso = DPSO(graph.number_of_nodes(), population_size, iterations, 0.30, 0.50, 0.30, fitness_func)

        print("> RUNNING GENETIC ALGORITHM")
        ga_start_time = time.time()
        ga.run()
        ga_end_time = time.time()
        ga_run_time = ga_end_time - ga_start_time
        print(f"> GENETIC ALGORITHM COMPLETE\nRUNTIME: {ga_run_time} SECONDS\n"
              f"BEST FITNESS: {ga.best_fitness_value_oat}")

        print("-------------------------------------------------------------------------")

        print("> RUNNING DISCRETE PARTICLE OPTIMIZATION ALGORITHM")
        dpso_start_time = time.time()
        dpso.run()
        dpso_end_time = time.time()
        dpso_run_time = dpso_end_time - dpso_start_time
        print(f"> DISCRETE PARTICLE OPTIMIZATION ALGORITHM COMPLETE\nRUNTIME: {dpso_run_time} SECONDS\n"
              f"BEST FITNESS: {dpso.global_best_fitness}")

        print("=============================================================================")

        runtime_ga.append(ga_run_time)
        runtime_dpso.append(dpso_run_time)

        results_ga.append(ga.best_fitness_value_oat)
        results_dpso.append(dpso.global_best_fitness)

    compare_metrics(runtime_ga, runtime_dpso, "GA", "DPSO", "GA vs DPSO (Runtime)")
    compare_metrics(results_ga, results_dpso, "GA", "DPSO", "GA vs DPSO (Solution)")
