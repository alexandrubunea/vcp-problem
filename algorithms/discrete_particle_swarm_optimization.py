from typing import Callable

import numpy as np


class DPSO:
    def __init__(self, solution_size: int, population_size: int, max_iterations: int, c1: float, c2: float, w: float,
                 fitness_function: Callable[[np.ndarray], int]):
        # Default values
        self.particle = np.empty(0)
        self.velocity = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.global_best = None
        self.global_best_fitness = None

        # Assign args
        self.solution_size = solution_size
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.c1 = c1  # Acceleration positive constant, controlls how much does a particle relay on own experience
        self.c2 = c2  # Acceleration positive constant, controlls how much does a particle relay on global experience
        self.w = w  # Inertia factor, controlls how much does the previous velocity influences the current one
        self.fitness_function = fitness_function

        # Init the population
        self.__initialize_population__()

        # Metrics to build graphs
        self.avg_fitness_evolution = []
        self.best_fitness_evolution = []
        self.best_particle_evolution = []

    def run(self):
        for iteration in range(self.max_iterations):
            if not iteration % 10:
                print(f"Current iteration {iteration}/{self.max_iterations}")

            fitness_iteration = []

            for i in range(self.population_size):
                # We need two random numbers between [0, 1] for every position of the particle
                # don't forget this is a solution for a discrete problem so the position of the particle
                # is a binary sequence, something like [1,0,0,1,1] and we need a random r1, r2 for each one
                r1, r2 = np.random.random(self.solution_size), np.random.random(self.solution_size)

                self.velocity[i] = (self.w * self.velocity[i] +  # Current velocity
                                    self.c1 * r1 * (self.personal_best[i] - self.velocity[i]) +  # Move towards BP
                                    self.c2 * r2 * (self.global_best - self.velocity[i]))  # Move towards GB

                # We need to convert our result to a binary one, so what we will need to do? Well we need to use
                # a sigmoid function that will take a continuous value and convert it into a value between [0, 1]
                sigmoid = 1 / (1 + np.exp(-self.velocity[i]))  # Oh, and always watch out numpy, very powerful

                # Now, in a perfect world (not in the discrete realm) we wouldn't need the sigmoid, because we could
                # just add the velocity to the current position and obtain the new position, but right now we are not
                # in a perfect world, so why we need to convert the velocity into a value between [0, 1]? We will use
                # the velocity as a probability, not a position on a plane (that's why it's a discrete problem) and
                # will simply check if a random number that we generate respects the probability then in our binary
                # sequence (or combinatorial sequence) we will set the value to 1, or 0 otherwise.
                new_position = (np.random.random(self.solution_size) < sigmoid).astype(int)
                self.particle[i] = new_position  # We update the particle's position

                # Calculate the fitness and update the bests
                fitness = self.fitness_function(self.particle[i])
                fitness_iteration.append(fitness)

                if fitness > self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.particle[i].copy()

                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.particle[i].copy()

            self.avg_fitness_evolution.append(np.average(fitness_iteration))
            self.best_fitness_evolution.append(self.global_best_fitness)
            self.best_particle_evolution.append(self.global_best[:])

    def __initialize_population__(self):
        self.particle = np.random.randint(2, size=(self.population_size, self.solution_size))
        self.velocity = np.random.random((self.population_size, self.solution_size))

        self.personal_best = np.copy(self.particle)
        self.personal_best_fitness = np.array([self.fitness_function(particle) for particle in self.particle])
        self.global_best = self.personal_best[np.argmax(self.personal_best_fitness)]
        self.global_best_fitness = max(self.personal_best_fitness)
