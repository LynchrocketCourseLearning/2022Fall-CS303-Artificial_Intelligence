import functools
import random
from abc import abstractmethod, ABC

import numpy as np

from utils import *

def mutate(chromosome):
    n = len(chromosome)
    point1 = random.randrange(0, n)
    point2 = point1
    while point2 == point1:
        point2 = random.randrange(0, n)
    c = chromosome.copy()
    temp = c[point1]
    c[point1] = c[point2]
    c[point2] = temp
    return c


def crossover(chromosome1, chromosome2):
    n = len(chromosome1)
    c1 = []
    c2 = []
    point = random.randrange(0, n)

    for i in range(point):
        c1.append(chromosome2[i])
        c2.append(chromosome1[i])

    for j in range(n):
        if chromosome1[j] not in c1:
            c1.append(chromosome1[j])
    for k in range(n):
        if chromosome2[k] not in c2:
            c2.append(chromosome2[k])
    return c1, c2


def select(r, population, fitness):
    p = population.copy()

    def compare(a, b):
        return fitness(a) - fitness(b)

    p.sort(key=functools.cmp_to_key(compare))
    return p[0:r]


class GAProblem(ABC):
    @abstractmethod
    def init_population(self, pop_size): pass

    @abstractmethod
    def fitness(self, sample): pass

    @abstractmethod
    def reproduce(self, population): pass

    @abstractmethod
    def replacement(self, old, new): pass


class NQueensProblem(GAProblem):
    def __init__(self, n):
        self.n = n
        self.max_fitness = n * (n - 1) // 2  # max number if non-attacking pairs

    def init_population(self, pop_size):
        # TODO:alomost the same as the previous problem.
        population = []
        initial = []
        for i in range(self.n):
            initial.append(i)
        for i in range(pop_size):
            individual = mutate(initial)
            population.append(individual)

        return population

    def fitness(self, queens):
        """
        TODO
        hint: count the non-attacking pairs
        """
        sum = 0
        for i in range(len(queens)):
            for j in range(len(queens)):
                if not self.isConflict((i, queens[i]), (j, queens[j])):
                    sum = sum + 1
        return sum

    def isConflict(self, a, b):
        if a[0] + a[1] == b[0] + b[1]:
            return True
        if a[0] - a[1] == b[0] - b[1]:
            return True
        if a[0] == b[0] or a[1] == b[1]:
            return True
        return False

    def reproduce(self, population, mutation_rate):
        # TODO:almost the same as the previous problem.
        p = select(12, population, fitness=self.fitness)
        for i in range(3):
            for j in range(i+1, 3):
                x, y = crossover(p[i], p[j])
                p.append(x)
                p.append(y)
        for i in range(len(population) - len(p)):
            p.append(mutate(population[i]))
        return p

    def replacement(self, old, new):
        """
        You can use your own strategy, for example retain some solutions from the old population
        """
        return new

    def __repr__(self):
        return f"{self.n}-Queens Problem"


def genetic_algorithm(
        problem: NQueensProblem,
        ngen, n_init_size, mutation_rate,
        log_intervel=100
):
    population = problem.init_population(n_init_size)
    best = max(population, key=problem.fitness)
    history = [(0, list(map(problem.fitness, population)))]

    for gen in range(ngen):
        next_gen = problem.reproduce(population, mutation_rate)
        population = problem.replacement(population, next_gen)

        if gen % log_intervel == 0:
            current_best = max(population, key=problem.fitness)
            if problem.fitness(current_best) > problem.fitness(best): best = current_best
            print(f"Generation: {gen}/{ngen},\tBest: {best},\tFitness={problem.fitness(best)}")
            history.append((gen, list(map(problem.fitness, population))))

    history.append((ngen - 1, list(map(problem.fitness, population))))
    return best, history


if __name__ == "__main__":
    ngen = 2000
    init_size = 120
    mutation_rate = 0.08

    n = 8
    problem = NQueensProblem(n)
    solution, history = genetic_algorithm(problem, ngen, init_size, mutation_rate)

# Example of how to use this function
# plot_NQueens([4, 2, 0, 6, 1, 7, 5, 3])
# replace the parameter with your own results
    plot_NQueens(solution)

# Visualize the evolution of the polulation
    bins = np.linspace(0, problem.max_fitness, problem.max_fitness)
    plot_evolution(history, bins)
